//! Adapter checkpointing: persist and restore the trainable `LoRA` parameters.
//!
//! A training run's learnable state is exactly the set of [`Var`]s the optimizer
//! updates — the `LoRA` `A`/`B` factors exposed by [`crate::Policy::trainable_vars`].
//! [`save_adapter`] writes them to `adapter.safetensors` (plus a small
//! `manifest.json`) under a checkpoint directory; [`load_adapter`] reads them back
//! into a live model's `Var`s.
//!
//! ## Generic over the policy
//!
//! Save/load operate on a `&[Var]` slice, not a concrete model. A cloned [`Var`]
//! shares its inner storage with the original (see
//! [`crate::Policy::trainable_vars`]), so [`load_adapter`] calls [`Var::set`] on
//! the slice and the change is reflected in whatever model those `Var`s came from.
//! The *same* code therefore checkpoints the toy policy and the Qwen policy. The
//! slice order is the model's stable `trainable_vars()` order; load validates the
//! tensor count and each tensor's shape and dtype against the live `Var`s, so a
//! checkpoint from a mismatched architecture fails loud rather than corrupting the
//! model.
//!
//! ## What is NOT persisted (yet)
//!
//! Only the adapter weights and the completed-step count. The `AdamW` optimizer
//! moments and the rollout sampler RNG are intentionally **not** serialized
//! (candle exposes neither through a public accessor): a resumed run restores the
//! trained weights and the step index but re-warms Adam's bias correction and
//! re-seeds sampling. This is a documented limitation — bit-exact,
//! momentum-faithful resume is deferred to P5 (where the optimizer state would be
//! owned/extended). The adapter weights themselves round-trip bit-exactly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor, Var};
use serde::{Deserialize, Serialize};

/// Filename of the serialized adapter tensors within a checkpoint directory.
const ADAPTER_FILE: &str = "adapter.safetensors";
/// Filename of the checkpoint manifest within a checkpoint directory.
const MANIFEST_FILE: &str = "manifest.json";
/// On-disk checkpoint layout version; bumped on an incompatible format change.
const FORMAT_VERSION: u32 = 1;

/// Errors raised while saving or loading an adapter checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// A filesystem operation (create dir / read / write) failed.
    #[error("checkpoint io error at {path}: {source}")]
    Io {
        /// Path being operated on when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// A candle tensor or safetensors operation failed.
    #[error("checkpoint tensor error: {0}")]
    Candle(#[from] candle_core::Error),
    /// The manifest could not be serialized or deserialized.
    #[error("checkpoint manifest error: {0}")]
    Manifest(#[from] serde_json::Error),
    /// The on-disk checkpoint does not match the model it is being loaded into:
    /// an unknown format version, a differing tensor count, a missing tensor, or a
    /// shape/dtype mismatch against the live trainable [`Var`]s.
    #[error("checkpoint mismatch: {0}")]
    Mismatch(String),
}

/// Self-describing metadata stored alongside the adapter tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointManifest {
    /// On-disk format version (validated by [`load_adapter`]).
    pub format_version: u32,
    /// Number of optimizer steps completed before this checkpoint was written —
    /// the step index a resumed run should continue from
    /// (see [`crate::Trainer::train_from`]).
    pub step: u64,
    /// Number of trainable tensors persisted (the `trainable_vars()` count).
    pub num_vars: usize,
}

/// Tensor key for the `i`-th trainable var, zero-padded so lexical order matches
/// numeric order.
fn var_key(i: usize) -> String {
    format!("lora.{i:05}")
}

/// Build a [`CheckpointError::Io`] for `path`.
fn io(path: impl Into<PathBuf>, source: std::io::Error) -> CheckpointError {
    CheckpointError::Io {
        path: path.into(),
        source,
    }
}

/// Persist `vars` (the trainable `LoRA` factors) and a manifest to `dir`.
///
/// Writes `dir/adapter.safetensors` and `dir/manifest.json`, creating `dir` (and
/// parents) if needed. `step` is recorded as the number of completed optimizer
/// steps — the index a resume should continue from. Each tensor is moved to the
/// CPU and made contiguous before serialization, so this works for vars living on
/// any device.
///
/// The adapter tensors are written **first** and the manifest **last**, and
/// [`load_adapter`] reads the manifest first — so a crash mid-write leaves the
/// manifest absent and the load fails cleanly rather than reading a truncated
/// adapter. Each checkpoint lives in its own directory, so a failed (or re-run)
/// write never corrupts a *prior* checkpoint. (The two writes are not atomic as a
/// unit; a fully crash-safe temp-dir+rename is a possible later hardening.)
///
/// # Errors
///
/// Returns [`CheckpointError`] if `dir` cannot be created, a tensor cannot be
/// moved to CPU / serialized, or the manifest cannot be written.
pub fn save_adapter(dir: impl AsRef<Path>, vars: &[Var], step: u64) -> Result<(), CheckpointError> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| io(dir, e))?;

    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(vars.len());
    for (i, v) in vars.iter().enumerate() {
        // CPU + contiguous so a CUDA-resident adapter serializes the same way.
        let t = v.as_tensor().to_device(&Device::Cpu)?.contiguous()?;
        tensors.insert(var_key(i), t);
    }
    let adapter_path = dir.join(ADAPTER_FILE);
    candle_core::safetensors::save(&tensors, &adapter_path)?;

    let manifest = CheckpointManifest {
        format_version: FORMAT_VERSION,
        step,
        num_vars: vars.len(),
    };
    let manifest_path = dir.join(MANIFEST_FILE);
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json).map_err(|e| io(&manifest_path, e))?;
    Ok(())
}

/// Restore a checkpoint from `dir` into `vars`, in place.
///
/// Reads `dir/manifest.json` and `dir/adapter.safetensors`, validates that the
/// checkpoint matches `vars` (format version, tensor count, and each tensor's
/// shape and dtype), then calls [`Var::set`] on each — updating the model those
/// `Var`s belong to (cloned vars share storage). Returns the manifest; its `step`
/// is where a resumed run should continue.
///
/// # Errors
///
/// Returns [`CheckpointError::Mismatch`] if the format version is unknown, the
/// tensor count differs, a tensor is missing, or any tensor's shape/dtype does not
/// match the corresponding live `Var`; or [`CheckpointError::Io`] /
/// [`CheckpointError::Candle`] / [`CheckpointError::Manifest`] on read failures.
pub fn load_adapter(
    dir: impl AsRef<Path>,
    vars: &[Var],
) -> Result<CheckpointManifest, CheckpointError> {
    let dir = dir.as_ref();

    let manifest_path = dir.join(MANIFEST_FILE);
    let raw = std::fs::read_to_string(&manifest_path).map_err(|e| io(&manifest_path, e))?;
    let manifest: CheckpointManifest = serde_json::from_str(&raw)?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(CheckpointError::Mismatch(format!(
            "unknown checkpoint format_version {} (this build reads {FORMAT_VERSION})",
            manifest.format_version
        )));
    }
    if manifest.num_vars != vars.len() {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint has {} tensors but the model exposes {} trainable vars",
            manifest.num_vars,
            vars.len()
        )));
    }

    let adapter_path = dir.join(ADAPTER_FILE);
    let loaded = candle_core::safetensors::load(&adapter_path, &Device::Cpu)?;
    for (i, v) in vars.iter().enumerate() {
        let key = var_key(i);
        let t = loaded.get(&key).ok_or_else(|| {
            CheckpointError::Mismatch(format!("checkpoint is missing tensor {key}"))
        })?;
        let want = v.as_tensor();
        if t.dims() != want.dims() {
            return Err(CheckpointError::Mismatch(format!(
                "tensor {key}: checkpoint shape {:?} != model shape {:?}",
                t.dims(),
                want.dims()
            )));
        }
        if t.dtype() != want.dtype() {
            return Err(CheckpointError::Mismatch(format!(
                "tensor {key}: checkpoint dtype {:?} != model dtype {:?}",
                t.dtype(),
                want.dtype()
            )));
        }
        // `Var::set` requires the *destination* var be contiguous (freshly
        // allocated LoRA factors are) and copies from a same-device source. Move the
        // CPU-loaded tensor onto the var's device; shape and dtype are already
        // validated equal. (`contiguous()` on the source is belt-and-suspenders.)
        let t = t.to_device(want.device())?.contiguous()?;
        v.set(&t)?;
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

    /// A unique temp directory, removed on drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            p.push(format!("ferrl-ckpt-{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Two distinct vars of different shapes (mirrors a `LoRA` A `[rank,in]` and
    /// B `[out,rank]`), filled deterministically.
    fn make_vars() -> Vec<Var> {
        let a = Tensor::from_vec(
            (0..6).map(|i| i as f32).collect::<Vec<_>>(),
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let b = Tensor::from_vec(
            (0..8).map(|i| i as f32 * -0.5).collect::<Vec<_>>(),
            (4, 2),
            &Device::Cpu,
        )
        .unwrap();
        vec![Var::from_tensor(&a).unwrap(), Var::from_tensor(&b).unwrap()]
    }

    fn snapshot(vars: &[Var]) -> Vec<Vec<f32>> {
        vars.iter()
            .map(|v| {
                v.as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn save_then_load_round_trips_bit_exactly() {
        let tmp = TempDir::new("roundtrip");
        let vars = make_vars();
        let original = snapshot(&vars);
        save_adapter(tmp.path(), &vars, 7).unwrap();

        // Clobber the vars, then load them back.
        for v in &vars {
            let z = Tensor::zeros(v.as_tensor().dims(), DType::F32, &Device::Cpu).unwrap();
            v.set(&z).unwrap();
        }
        assert_ne!(snapshot(&vars), original, "clobber did not change the vars");

        let manifest = load_adapter(tmp.path(), &vars).unwrap();
        assert_eq!(manifest.step, 7);
        assert_eq!(manifest.num_vars, 2);
        assert_eq!(
            snapshot(&vars),
            original,
            "loaded adapter must equal the saved one bit-for-bit"
        );
    }

    #[test]
    fn load_writes_through_to_a_fresh_set_of_aliasing_vars() {
        // The save and load var slices are different Var instances of the same
        // shapes — load must populate the second slice (this is the resume/eval
        // path: a fresh model's trainable_vars()).
        let tmp = TempDir::new("alias");
        let src = make_vars();
        let saved = snapshot(&src);
        save_adapter(tmp.path(), &src, 3).unwrap();

        let dst = make_vars();
        for v in &dst {
            v.set(&Tensor::ones(v.as_tensor().dims(), DType::F32, &Device::Cpu).unwrap())
                .unwrap();
        }
        load_adapter(tmp.path(), &dst).unwrap();
        assert_eq!(snapshot(&dst), saved);
    }

    #[test]
    fn load_rejects_wrong_var_count() {
        let tmp = TempDir::new("count");
        let vars = make_vars();
        save_adapter(tmp.path(), &vars, 1).unwrap();
        let just_one = vec![vars[0].clone()];
        let err = load_adapter(tmp.path(), &just_one).unwrap_err();
        assert!(matches!(err, CheckpointError::Mismatch(_)), "got {err:?}");
    }

    #[test]
    fn load_rejects_shape_mismatch() {
        let tmp = TempDir::new("shape");
        let vars = make_vars();
        save_adapter(tmp.path(), &vars, 1).unwrap();
        // Same count, but the first var has a different shape.
        let bad = vec![
            Var::from_tensor(&Tensor::zeros((3, 3), DType::F32, &Device::Cpu).unwrap()).unwrap(),
            vars[1].clone(),
        ];
        let err = load_adapter(tmp.path(), &bad).unwrap_err();
        match err {
            CheckpointError::Mismatch(m) => assert!(m.contains("shape"), "{m}"),
            other => panic!("expected shape mismatch, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_dtype_mismatch() {
        let tmp = TempDir::new("dtype");
        let vars = make_vars(); // F32
        save_adapter(tmp.path(), &vars, 1).unwrap();
        // Same count and shapes, but the first target var is F64.
        let a64 = Tensor::zeros((2, 3), DType::F64, &Device::Cpu).unwrap();
        let bad = vec![Var::from_tensor(&a64).unwrap(), vars[1].clone()];
        let err = load_adapter(tmp.path(), &bad).unwrap_err();
        match err {
            CheckpointError::Mismatch(m) => assert!(m.contains("dtype"), "{m}"),
            other => panic!("expected dtype mismatch, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_unknown_format_version() {
        let tmp = TempDir::new("version");
        let vars = make_vars();
        save_adapter(tmp.path(), &vars, 1).unwrap();
        // Rewrite the manifest with a future format version.
        let bumped = CheckpointManifest {
            format_version: FORMAT_VERSION + 1,
            step: 1,
            num_vars: vars.len(),
        };
        std::fs::write(
            tmp.path().join(MANIFEST_FILE),
            serde_json::to_string(&bumped).unwrap(),
        )
        .unwrap();
        let err = load_adapter(tmp.path(), &vars).unwrap_err();
        match err {
            CheckpointError::Mismatch(m) => assert!(m.contains("format_version"), "{m}"),
            other => panic!("expected version mismatch, got {other:?}"),
        }
    }

    #[test]
    fn load_missing_dir_is_an_io_error() {
        let tmp = TempDir::new("missing");
        let vars = make_vars();
        let err = load_adapter(tmp.path().join("nope"), &vars).unwrap_err();
        assert!(matches!(err, CheckpointError::Io { .. }), "got {err:?}");
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let m = CheckpointManifest {
            format_version: FORMAT_VERSION,
            step: 42,
            num_vars: 8,
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: CheckpointManifest = serde_json::from_str(&j).unwrap();
        assert_eq!(back, m);
    }
}
