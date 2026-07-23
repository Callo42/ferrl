//! Helpers for onboarding a real Hugging Face checkpoint.
//!
//! Two small conveniences that every real-model run otherwise hand-rolls:
//!
//! - [`eos_from_config`] reads a checkpoint's end-of-sequence token id straight
//!   from its `config.json` (candle's model `Config` types do not deserialize
//!   `eos_token_id`), to feed
//!   [`TrainerConfig::eos_token_id`](crate::TrainerConfig::eos_token_id).
//! - [`chatml`] frames a prompt in the **`ChatML`** chat template the Qwen-family
//!   instruct models expect, for runs against an `-Instruct` checkpoint rather than
//!   a base model.
//!
//! This is deliberately *not* a general Jinja `chat_template` engine: ferrl targets
//! the Qwen3.5/3.6 family, whose instruct variants use `ChatML`, so a single
//! hand-written framing covers the supported models without a templating
//! dependency. A model with a different chat format should build its prompt itself.

use std::fs;
use std::path::{Path, PathBuf};

/// An error reading a checkpoint's `config.json`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HfError {
    /// `config.json` could not be read.
    #[error("failed to read {path}")]
    Io {
        /// The path that could not be read.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// `config.json` could not be parsed as JSON.
    #[error("failed to parse {path} as JSON")]
    Parse {
        /// The path that failed to parse.
        path: PathBuf,
        /// The underlying deserialization error.
        #[source]
        source: serde_json::Error,
    },
    /// `eos_token_id` exists but is not a supported scalar/list of token ids.
    #[error("invalid eos_token_id in {path}: {reason}")]
    InvalidEos {
        /// The checkpoint config containing the invalid field.
        path: PathBuf,
        /// The exact shape/range failure.
        reason: String,
    },
    /// A scalar-only caller encountered several declared EOS ids.
    #[error("ambiguous multi-token eos_token_id in {path}: {token_ids:?}")]
    AmbiguousEos {
        /// The checkpoint config containing the ambiguous field.
        path: PathBuf,
        /// The distinct declared EOS ids, in checkpoint order.
        token_ids: Vec<u32>,
    },
    /// `vocab_size` is absent or not a positive integer representable as `usize`.
    #[error("invalid vocab_size in {path}: {reason}")]
    InvalidVocabSize {
        /// The checkpoint config containing the invalid field.
        path: PathBuf,
        /// The exact shape/range failure.
        reason: String,
    },
    /// A requested EOS mode cannot be resolved against checkpoint/model/tokenizer metadata.
    #[error("invalid EOS selection for {path}: {reason}")]
    InvalidEosSelection {
        /// The checkpoint whose EOS semantics were selected.
        path: PathBuf,
        /// The exact missing, ambiguous, membership, or vocabulary failure.
        reason: String,
    },
    /// Distributed ranks did not resolve identical EOS semantics.
    #[error("resolved EOS consensus failed: {reason}")]
    EosConsensus {
        /// The collective or rank-disagreement detail.
        reason: String,
    },
}

/// Validated EOS metadata declared by a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointEos {
    /// Neither `text_config.eos_token_id` nor top-level `eos_token_id` exists.
    Missing,
    /// Exactly one EOS id is declared.
    Single(u32),
    /// Several distinct EOS ids are declared and require an explicit selection.
    Multiple(Vec<u32>),
}

/// Public selection mode for checkpoint-backed generation EOS semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointEosSelection {
    /// Resolve the checkpoint's one declared scalar EOS id.
    CheckpointDefault,
    /// Supply or override one explicit id, validating multi-EOS membership when applicable.
    Explicit(u32),
    /// Deliberately generate full-width continuations without EOS retirement.
    Disabled,
}

/// Read the model's `eos_token_id` from `<dir>/config.json`.
///
/// Looks under the multimodal wrapper's `text_config` first (the Qwen3.5/3.6
/// checkpoint shape), then the top level — so both a flat text-only config and a
/// wrapped one resolve. Returns `Ok(None)` only when the field is absent. Invalid
/// metadata and list-valued multi-EOS declarations fail closed; callers that need
/// to validate an explicit member selection can use
/// [`checkpoint_eos_from_config`].
///
/// # Errors
///
/// Returns [`HfError::Io`] if `<dir>/config.json` cannot be read,
/// [`HfError::Parse`] if it is not valid JSON, [`HfError::InvalidEos`] for an
/// invalid declaration, or [`HfError::AmbiguousEos`] when several ids are
/// declared.
pub fn eos_from_config<P: AsRef<Path>>(dir: P) -> Result<Option<u32>, HfError> {
    let path = dir.as_ref().join("config.json");
    match checkpoint_eos_from_config(dir)? {
        CheckpointEos::Missing => Ok(None),
        CheckpointEos::Single(id) => Ok(Some(id)),
        CheckpointEos::Multiple(token_ids) => Err(HfError::AmbiguousEos { path, token_ids }),
    }
}

/// Read and validate all EOS ids declared by `<dir>/config.json`.
///
/// A nested `text_config.eos_token_id` is authoritative when present. Lists must
/// be non-empty, contain distinct `u32` ids, and become [`CheckpointEos::Multiple`]
/// when they contain more than one entry.
///
/// # Errors
///
/// Returns [`HfError::InvalidEos`] for malformed, out-of-range, empty, or
/// duplicate declarations, in addition to the file/JSON failures documented by
/// [`eos_from_config`].
pub fn checkpoint_eos_from_config<P: AsRef<Path>>(dir: P) -> Result<CheckpointEos, HfError> {
    let (path, json) = read_checkpoint_json(dir.as_ref())?;
    eos_from_json(&json).map_err(|reason| HfError::InvalidEos { path, reason })
}

/// Read the effective model `vocab_size` from `<dir>/config.json`.
///
/// The multimodal `text_config.vocab_size` field is authoritative when present;
/// otherwise the top-level value is used.
///
/// # Errors
///
/// Returns [`HfError::InvalidVocabSize`] when the value is missing, zero,
/// malformed, or too large for this process.
pub fn vocab_size_from_config<P: AsRef<Path>>(dir: P) -> Result<usize, HfError> {
    let (path, json) = read_checkpoint_json(dir.as_ref())?;
    vocab_size_from_json(&json).map_err(|reason| HfError::InvalidVocabSize { path, reason })
}

/// Resolve one generation EOS mode against checkpoint, model, and tokenizer metadata.
///
/// Checkpoint-default mode requires exactly one declared EOS id. An explicit id may
/// supply EOS when the checkpoint declaration is missing or override one scalar
/// declaration. For a multi-EOS declaration, the explicit id must be a declared
/// member. Every enabled id must fit the model vocabulary and exist in the loaded
/// tokenizer; sparse tokenizer ids are accepted.
/// [`CheckpointEosSelection::Disabled`] is the only mode that returns `None`.
///
/// # Errors
///
/// Returns the checkpoint parsing errors documented by
/// [`checkpoint_eos_from_config`], or [`HfError::InvalidEosSelection`] when
/// checkpoint-default mode is missing or ambiguous, an explicit multi-EOS selection
/// is not a declared member, or the selected id is outside the effective model or
/// tokenizer vocabulary.
pub fn resolve_checkpoint_eos<P: AsRef<Path>>(
    dir: P,
    tokenizer: &crate::HfTokenizer,
    selection: CheckpointEosSelection,
) -> Result<Option<u32>, HfError> {
    if selection == CheckpointEosSelection::Disabled {
        return Ok(None);
    }
    let dir = dir.as_ref();
    let path = dir.join("config.json");
    let checkpoint = checkpoint_eos_from_config(dir)?;
    let selected = match (selection, checkpoint) {
        (CheckpointEosSelection::CheckpointDefault, CheckpointEos::Missing) => {
            return Err(HfError::InvalidEosSelection {
                path,
                reason: "checkpoint declares no eos_token_id; select one explicit id or disable EOS explicitly".into(),
            });
        }
        (CheckpointEosSelection::CheckpointDefault, CheckpointEos::Multiple(ids)) => {
            return Err(HfError::InvalidEosSelection {
                path,
                reason: format!(
                    "checkpoint declares multiple EOS ids {ids:?}; select one declared id explicitly or disable EOS explicitly"
                ),
            });
        }
        (CheckpointEosSelection::Explicit(id), CheckpointEos::Multiple(ids)) => {
            if !ids.contains(&id) {
                return Err(HfError::InvalidEosSelection {
                    path,
                    reason: format!(
                        "explicit EOS id {id} is not a member of the checkpoint's declared EOS ids {ids:?}"
                    ),
                });
            }
            id
        }
        (CheckpointEosSelection::CheckpointDefault, CheckpointEos::Single(id))
        | (CheckpointEosSelection::Explicit(id), _) => id,
        (CheckpointEosSelection::Disabled, _) => unreachable!("disabled EOS returned above"),
    };
    let model_vocab = vocab_size_from_config(dir)?;
    let selected_index = usize::try_from(selected).map_err(|_| HfError::InvalidEosSelection {
        path: path.clone(),
        reason: format!("EOS id {selected} does not fit usize"),
    })?;
    if selected_index >= model_vocab {
        return Err(HfError::InvalidEosSelection {
            path,
            reason: format!("EOS id {selected} is outside model vocab_size {model_vocab}"),
        });
    }
    if !tokenizer.contains_id(selected) {
        return Err(HfError::InvalidEosSelection {
            path,
            reason: format!(
                "EOS id {selected} is not present in the loaded tokenizer vocabulary (token count {})",
                tokenizer.vocab_size()
            ),
        });
    }
    Ok(Some(selected))
}

/// Require every rank in `comm` to have resolved identical EOS semantics.
///
/// The five-byte comparison distinguishes `None` from every `Some(id)`, including
/// `Some(0)`. Callers must coordinate rank-local checkpoint/tokenizer resolution
/// failures before entering this collective.
///
/// # Errors
///
/// Returns [`HfError::EosConsensus`] on a collective failure or rank disagreement.
pub fn validate_resolved_eos_consensus(
    eos_token_id: Option<u32>,
    comm: &dyn crate::Comm,
) -> Result<(), HfError> {
    if comm.world_size() <= 1 {
        return Ok(());
    }
    let mut bytes = [0_u8; 5];
    if let Some(eos_token_id) = eos_token_id {
        bytes[0] = 1;
        bytes[1..].copy_from_slice(&eos_token_id.to_le_bytes());
    }
    let mut mismatch = false;
    for byte in bytes {
        let value = f64::from(byte);
        let canonical = comm
            .all_reduce_scalar_sum(if comm.rank() == 0 { value } else { 0.0 })
            .map_err(|error| HfError::EosConsensus {
                reason: error.to_string(),
            })?;
        mismatch |= canonical != value;
    }
    let mismatched_ranks = comm
        .all_reduce_scalar_sum(if mismatch { 1.0 } else { 0.0 })
        .map_err(|error| HfError::EosConsensus {
            reason: error.to_string(),
        })?;
    if mismatched_ranks != 0.0 {
        return Err(HfError::EosConsensus {
            reason:
                "ranks resolved different EOS token semantics from checkpoint/tokenizer metadata"
                    .into(),
        });
    }
    Ok(())
}

fn read_checkpoint_json(dir: &Path) -> Result<(PathBuf, serde_json::Value), HfError> {
    let path = dir.join("config.json");
    let bytes = fs::read(&path).map_err(|source| HfError::Io {
        path: path.clone(),
        source,
    })?;
    let json = serde_json::from_slice(&bytes).map_err(|source| HfError::Parse {
        path: path.clone(),
        source,
    })?;
    Ok((path, json))
}

fn preferred_field<'a>(json: &'a serde_json::Value, field: &str) -> Option<&'a serde_json::Value> {
    json.get("text_config")
        .and_then(|text| text.get(field))
        .or_else(|| json.get(field))
}

fn eos_from_json(json: &serde_json::Value) -> Result<CheckpointEos, String> {
    let Some(raw) = preferred_field(json, "eos_token_id") else {
        return Ok(CheckpointEos::Missing);
    };
    if let Some(id) = parse_token_id(raw) {
        return Ok(CheckpointEos::Single(id));
    }
    let Some(values) = raw.as_array() else {
        return Err("expected a non-negative integer or a non-empty integer list".into());
    };
    if values.is_empty() {
        return Err("EOS token list must not be empty".into());
    }
    let mut ids = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let id = parse_token_id(value)
            .ok_or_else(|| format!("EOS token list entry {index} is not a u32 integer"))?;
        if ids.contains(&id) {
            return Err(format!("EOS token list contains duplicate id {id}"));
        }
        ids.push(id);
    }
    if ids.len() == 1 {
        Ok(CheckpointEos::Single(ids[0]))
    } else {
        Ok(CheckpointEos::Multiple(ids))
    }
}

fn parse_token_id(value: &serde_json::Value) -> Option<u32> {
    value.as_u64().and_then(|id| u32::try_from(id).ok())
}

fn vocab_size_from_json(json: &serde_json::Value) -> Result<usize, String> {
    let raw = preferred_field(json, "vocab_size").ok_or_else(|| "field is missing".to_string())?;
    let value = raw
        .as_u64()
        .ok_or_else(|| "expected a positive integer".to_string())?;
    if value == 0 {
        return Err("must be >= 1".into());
    }
    usize::try_from(value).map_err(|_| "does not fit usize".into())
}

/// Frame a `user` turn (with an optional `system` turn) in the **`ChatML`** template.
///
/// Produces the prompt string an `-Instruct` Qwen model expects, ending with an
/// **open** assistant turn so the rollout continues as the assistant:
///
/// ```text
/// <|im_start|>system
/// {system}<|im_end|>
/// <|im_start|>user
/// {user}<|im_end|>
/// <|im_start|>assistant
/// ```
///
/// (The `system` block is omitted entirely when `system` is `None`.) The
/// `<|im_start|>` / `<|im_end|>` markers are the literal special tokens in a
/// Qwen instruct tokenizer's vocabulary, so [`HfTokenizer`](crate::HfTokenizer)
/// encodes them to their single token ids even though it adds no special tokens of
/// its own. Base models (no `ChatML` vocabulary) should not use this.
#[must_use]
pub fn chatml(system: Option<&str>, user: &str) -> String {
    let mut out = String::new();
    if let Some(system) = system {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user);
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Comm as _;

    struct TestCheckpoint(PathBuf);

    impl TestCheckpoint {
        fn new(tag: &str, config: &serde_json::Value) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("ferrl-hf-{tag}-{}-{nonce}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(
                path.join("config.json"),
                serde_json::to_vec(config).unwrap(),
            )
            .unwrap();
            std::fs::copy(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_tokenizer.json"),
                path.join("tokenizer.json"),
            )
            .unwrap();
            Self(path)
        }

        fn tokenizer(&self) -> crate::HfTokenizer {
            crate::HfTokenizer::from_file(self.0.join("tokenizer.json")).unwrap()
        }
    }

    impl Drop for TestCheckpoint {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn eos_from_json_reads_a_single_integer() {
        let json = serde_json::json!({ "eos_token_id": 151_645, "hidden_size": 16 });
        assert_eq!(eos_from_json(&json), Ok(CheckpointEos::Single(151_645)));
    }

    #[test]
    fn eos_from_json_is_none_when_absent() {
        let json = serde_json::json!({ "hidden_size": 16 });
        assert_eq!(eos_from_json(&json), Ok(CheckpointEos::Missing));
    }

    #[test]
    fn eos_from_json_prefers_the_nested_text_config() {
        // The Qwen3.5/3.6 multimodal-wrapper shape: EOS nested under text_config,
        // taken ahead of any top-level value.
        let json = serde_json::json!({
            "eos_token_id": 1,
            "text_config": { "eos_token_id": 151_645 },
        });
        assert_eq!(eos_from_json(&json), Ok(CheckpointEos::Single(151_645)));
    }

    #[test]
    fn eos_from_json_preserves_a_multi_eos_list_for_explicit_selection() {
        let json = serde_json::json!({ "eos_token_id": [151_645, 151_643] });
        assert_eq!(
            eos_from_json(&json),
            Ok(CheckpointEos::Multiple(vec![151_645, 151_643]))
        );
        assert_eq!(
            eos_from_json(&serde_json::json!({ "eos_token_id": [151_645] })),
            Ok(CheckpointEos::Single(151_645))
        );
    }

    #[test]
    fn eos_from_json_rejects_an_out_of_range_or_wrong_type() {
        for json in [
            serde_json::json!({ "eos_token_id": -1 }),
            serde_json::json!({ "eos_token_id": u64::from(u32::MAX) + 1 }),
            serde_json::json!({ "eos_token_id": "151645" }),
        ] {
            assert!(eos_from_json(&json).is_err());
        }
        for json in [
            serde_json::json!({ "eos_token_id": [] }),
            serde_json::json!({ "eos_token_id": [3, 3] }),
            serde_json::json!({ "eos_token_id": [3, -1] }),
            serde_json::json!({
                "eos_token_id": 3,
                "text_config": { "eos_token_id": null },
            }),
        ] {
            assert!(eos_from_json(&json).is_err());
        }
    }

    #[test]
    fn vocab_size_prefers_nested_and_rejects_invalid_values() {
        let nested = serde_json::json!({
            "vocab_size": 8,
            "text_config": { "vocab_size": 16 },
        });
        assert_eq!(vocab_size_from_json(&nested), Ok(16));
        for json in [
            serde_json::json!({}),
            serde_json::json!({ "vocab_size": 0 }),
            serde_json::json!({ "vocab_size": "16" }),
        ] {
            assert!(vocab_size_from_json(&json).is_err());
        }
    }

    #[test]
    fn eos_from_config_reads_a_committed_fixture() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/eos_config");
        assert_eq!(eos_from_config(dir).unwrap(), Some(151_645));
    }

    #[test]
    fn eos_from_config_errors_on_a_missing_config() {
        let err = eos_from_config("/no/such/checkpoint").unwrap_err();
        assert!(matches!(err, HfError::Io { .. }), "got {err:?}");
    }

    #[test]
    fn eos_from_config_rejects_ambiguous_or_malformed_metadata() {
        for fixture in ["eos_config_multi", "eos_config_malformed"] {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(fixture);
            assert!(eos_from_config(&dir).is_err(), "{fixture} must fail closed");
        }
    }

    #[test]
    fn strict_checkpoint_eos_resolution_requires_scalar_or_explicit_mode() {
        let missing = TestCheckpoint::new("missing-eos", &serde_json::json!({ "vocab_size": 4 }));
        assert!(resolve_checkpoint_eos(
            &missing.0,
            &missing.tokenizer(),
            CheckpointEosSelection::CheckpointDefault,
        )
        .is_err());
        assert_eq!(
            resolve_checkpoint_eos(
                &missing.0,
                &missing.tokenizer(),
                CheckpointEosSelection::Disabled,
            )
            .unwrap(),
            None
        );

        let multiple = TestCheckpoint::new(
            "multi-eos",
            &serde_json::json!({ "vocab_size": 4, "eos_token_id": [2, 3] }),
        );
        assert!(resolve_checkpoint_eos(
            &multiple.0,
            &multiple.tokenizer(),
            CheckpointEosSelection::CheckpointDefault,
        )
        .is_err());
        assert_eq!(
            resolve_checkpoint_eos(
                &multiple.0,
                &multiple.tokenizer(),
                CheckpointEosSelection::Explicit(3),
            )
            .unwrap(),
            Some(3)
        );
        assert!(resolve_checkpoint_eos(
            &multiple.0,
            &multiple.tokenizer(),
            CheckpointEosSelection::Explicit(1),
        )
        .is_err());
    }

    #[test]
    fn explicit_checkpoint_eos_can_supply_missing_or_override_scalar_declaration() {
        let missing = TestCheckpoint::new(
            "missing-eos-explicit-override",
            &serde_json::json!({ "vocab_size": 4 }),
        );
        assert_eq!(
            resolve_checkpoint_eos(
                &missing.0,
                &missing.tokenizer(),
                CheckpointEosSelection::Explicit(3),
            )
            .unwrap(),
            Some(3)
        );

        let scalar = TestCheckpoint::new(
            "scalar-eos-explicit-override",
            &serde_json::json!({ "vocab_size": 4, "eos_token_id": 2 }),
        );
        assert_eq!(
            resolve_checkpoint_eos(
                &scalar.0,
                &scalar.tokenizer(),
                CheckpointEosSelection::Explicit(3),
            )
            .unwrap(),
            Some(3)
        );
    }

    #[test]
    fn checkpoint_eos_rejects_model_valid_tokenizer_absent_id() {
        let checkpoint = TestCheckpoint::new(
            "tokenizer-membership",
            &serde_json::json!({ "vocab_size": 5, "eos_token_id": 4 }),
        );
        let tokenizer = checkpoint.tokenizer();
        assert!(!tokenizer.contains_id(4));

        let error = resolve_checkpoint_eos(
            &checkpoint.0,
            &tokenizer,
            CheckpointEosSelection::CheckpointDefault,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("EOS id 4 is not present in the loaded tokenizer vocabulary"),
            "{error}"
        );
    }

    #[test]
    fn checkpoint_eos_model_bound_is_independent_of_sparse_tokenizer_membership() {
        let checkpoint = TestCheckpoint::new(
            "model-bound",
            &serde_json::json!({ "vocab_size": 4, "eos_token_id": 4 }),
        );
        let tokenizer_path = checkpoint.0.join("tokenizer.json");
        let mut tokenizer_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&tokenizer_path).unwrap()).unwrap();
        tokenizer_json["model"]["vocab"]["<|special|>"] = serde_json::json!(4);
        tokenizer_json["added_tokens"][0]["id"] = serde_json::json!(4);
        std::fs::write(
            &tokenizer_path,
            serde_json::to_vec(&tokenizer_json).unwrap(),
        )
        .unwrap();
        let tokenizer = checkpoint.tokenizer();
        assert!(tokenizer.contains_id(4));

        let error = resolve_checkpoint_eos(
            &checkpoint.0,
            &tokenizer,
            CheckpointEosSelection::CheckpointDefault,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside model vocab_size 4"), "{error}");
    }

    #[test]
    fn resolved_eos_consensus_distinguishes_none_from_some_zero() {
        let results = std::thread::scope(|scope| {
            crate::LocalComm::world(2)
                .into_iter()
                .map(|comm| {
                    scope.spawn(move || {
                        let eos = if comm.rank() == 0 { Some(0) } else { None };
                        validate_resolved_eos_consensus(eos, &comm)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(Result::is_err), "{results:?}");
    }

    #[test]
    fn chatml_wraps_a_user_turn_and_opens_the_assistant_turn() {
        assert_eq!(
            chatml(None, "hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_includes_the_system_turn_when_present() {
        assert_eq!(
            chatml(Some("be terse"), "hi"),
            "<|im_start|>system\nbe terse<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
