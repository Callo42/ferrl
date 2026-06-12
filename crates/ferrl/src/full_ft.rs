//! Full fine-tuning support: the **`Var`-registry `VarBuilder` backend**.
//!
//! `LoRA` is ferrl's default training mode; **full fine-tuning is the opt-in**
//! (the roadmap track locked 2026-06-12): every base weight a model fetches
//! through its [`VarBuilder`] is wrapped in a trainable [`Var`] at load time
//! and recorded in **get-order** — deterministic because it is the model's own
//! load order (layer-major), the same positional convention as the adapter
//! checkpoint contract. candle's own `VarMap` was rejected for this seam: its
//! `HashMap`-backed iteration order is nondeterministic, and the positional
//! checkpoint format needs a stable order, not names.
//!
//! Mechanics: the registering backend wraps the plain tensor-map backend;
//! every `get` fetches the raw tensor (shape/dtype/device checks included),
//! wraps it in a `Var` ([`Var::from_tensor`] copies to fresh storage — the
//! once-per-load RAM cost of trainability), caches it **by full name** (a
//! tied / re-fetched tensor must not mint two `Var`s), and returns the var's
//! **inner tensor** — same storage, same autograd id — so model forward code
//! is untouched and `backward` lands gradients on the registry's vars.
//!
//! The **`Var::set` visibility contract** this module leans on (verified at
//! candle 0.10.2 source; pinned by tests below): `Var::set` mutates the var's
//! storage **in place**, so every tensor that *shares* that storage (the
//! inner-tensor clones the backend hands out) sees optimizer updates — while
//! any tensor *derived* at load time (`cat`, `stack`, a real dtype cast) owns
//! fresh storage and would go silently stale after the first step. Loaders
//! must therefore store weights **as fetched**; the one structural exception
//! (packing per-expert `MoE` checkpoint tensors) excludes the raw fetches via
//! the backend's `exclude` rule and registers the *packed* tensor as a single
//! var through [`VarRegistry::register`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor, Var};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};

/// The ordered `(name, Var)` registry a full-fine-tuning load fills.
///
/// Cloning shares the registry (the backend and the loader hold the same
/// one). Registration order — the backend's get-order plus any explicit
/// [`register`](Self::register) calls, i.e. the model's load order — **is**
/// the positional checkpoint contract of a full-FT model.
#[derive(Debug, Clone, Default)]
pub(crate) struct VarRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

#[derive(Debug, Default)]
struct RegistryInner {
    /// Registration order — the positional contract.
    vars: Vec<(String, Var)>,
    /// Name → index into `vars` (the tied-weight dedup cache).
    by_name: HashMap<String, usize>,
}

impl VarRegistry {
    /// An empty registry.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Mint (or reuse, by `name`) the trainable [`Var`] for `tensor` and
    /// return its **inner tensor** — the storage-sharing handle the model
    /// stores. A repeated name returns the FIRST var's inner tensor (true
    /// weight tying); the duplicate registration is not recorded.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the var allocation fails or the registry
    /// lock is poisoned.
    pub(crate) fn register(&self, name: &str, tensor: &Tensor) -> CandleResult<Tensor> {
        let mut inner = self.lock()?;
        if let Some(&i) = inner.by_name.get(name) {
            return Ok(inner.vars[i].1.as_tensor().clone());
        }
        let var = Var::from_tensor(tensor)?;
        let out = var.as_tensor().clone();
        let i = inner.vars.len();
        inner.by_name.insert(name.to_owned(), i);
        inner.vars.push((name.to_owned(), var));
        Ok(out)
    }

    /// Every registered var, in registration (= load) order.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the registry lock is poisoned.
    pub(crate) fn vars(&self) -> CandleResult<Vec<Var>> {
        Ok(self.lock()?.vars.iter().map(|(_, v)| v.clone()).collect())
    }

    /// Registered names, in registration order (for the order/dedup gates).
    #[cfg(test)]
    pub(crate) fn names(&self) -> CandleResult<Vec<String>> {
        Ok(self.lock()?.vars.iter().map(|(n, _)| n.clone()).collect())
    }

    fn lock(&self) -> CandleResult<MutexGuard<'_, RegistryInner>> {
        self.inner.lock().map_err(|_| {
            candle_core::Error::Msg(
                "VarRegistry: poisoned lock (a previous load panicked mid-registration)"
                    .to_string(),
            )
        })
    }
}

/// The registering [`SimpleBackend`]: `inner` resolves the fetch, the
/// registry wraps the result in a var. Fetches whose **full dotted name**
/// matches `exclude` pass through RAW (no var minted) — the loader takes
/// responsibility for registering whatever it derives from them (the
/// packed-expert case).
struct VarRegistryBackend {
    inner: HashMap<String, Tensor>,
    registry: VarRegistry,
    exclude: fn(&str) -> bool,
}

impl SimpleBackend for VarRegistryBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> CandleResult<Tensor> {
        let t = SimpleBackend::get(&self.inner, s, name, h, dtype, dev)?;
        if (self.exclude)(name) {
            return Ok(t);
        }
        self.registry.register(name, &t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> CandleResult<Tensor> {
        let t = SimpleBackend::get_unchecked(&self.inner, name, dtype, dev)?;
        if (self.exclude)(name) {
            return Ok(t);
        }
        self.registry.register(name, &t)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        SimpleBackend::contains_tensor(&self.inner, name)
    }
}

/// Build the registry-backed [`VarBuilder`] over `tensors`, returning the
/// builder and the (initially empty) registry it fills. `exclude` names pass
/// through raw — see [`VarRegistryBackend`]'s contract above.
pub(crate) fn registry_varbuilder(
    tensors: HashMap<String, Tensor>,
    dtype: DType,
    device: &Device,
    exclude: fn(&str) -> bool,
) -> (VarBuilder<'static>, VarRegistry) {
    let registry = VarRegistry::new();
    let backend = VarRegistryBackend {
        inner: tensors,
        registry: registry.clone(),
        exclude,
    };
    let vb = VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
    (vb, registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::Cpu
    }

    fn map(names: &[&str]) -> HashMap<String, Tensor> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let t = Tensor::full((i + 1) as f32, (2, 3), &dev()).unwrap();
                ((*n).to_owned(), t)
            })
            .collect()
    }

    fn no_exclude(_: &str) -> bool {
        false
    }

    #[test]
    fn registration_order_is_get_order_not_map_order() {
        let (vb, reg) = registry_varbuilder(
            map(&["a.weight", "b.weight", "c.weight"]),
            DType::F32,
            &dev(),
            no_exclude,
        );
        // Fetch deliberately out of name order.
        vb.pp("c").get((2, 3), "weight").unwrap();
        vb.pp("a").get((2, 3), "weight").unwrap();
        vb.pp("b").get((2, 3), "weight").unwrap();
        assert_eq!(
            reg.names().unwrap(),
            vec!["c.weight", "a.weight", "b.weight"]
        );
        assert_eq!(reg.vars().unwrap().len(), 3);
    }

    #[test]
    fn a_refetched_name_reuses_the_var_instead_of_minting_two() {
        let (vb, reg) = registry_varbuilder(map(&["tied.weight"]), DType::F32, &dev(), no_exclude);
        let first = vb.pp("tied").get((2, 3), "weight").unwrap();
        let second = vb.pp("tied").get((2, 3), "weight").unwrap();
        let vars = reg.vars().unwrap();
        assert_eq!(vars.len(), 1, "a tied re-fetch must not mint a second var");
        // Both handles share the var's storage: a set is visible through each.
        let next = Tensor::full(9.0f32, (2, 3), &dev()).unwrap();
        vars[0].set(&next).unwrap();
        for (label, t) in [("first", &first), ("second", &second)] {
            let got = t.to_vec2::<f32>().unwrap();
            assert!(
                got.iter().flatten().all(|&v| v == 9.0),
                "{label} fetch does not see the var update: {got:?}"
            );
        }
    }

    /// The semantic pin the whole mode rests on: `Var::set` mutates storage
    /// in place, so the fetched (storage-sharing) tensor tracks updates while
    /// a load-time-derived copy goes stale.
    #[test]
    fn var_set_is_visible_through_the_fetched_tensor_but_not_a_derived_copy() {
        let (vb, reg) = registry_varbuilder(map(&["w.weight"]), DType::F32, &dev(), no_exclude);
        let fetched = vb.pp("w").get((2, 3), "weight").unwrap();
        let derived = fetched.copy().unwrap(); // a load-time derivation
        let next = Tensor::full(5.0f32, (2, 3), &dev()).unwrap();
        reg.vars().unwrap()[0].set(&next).unwrap();
        assert!(
            fetched
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .flatten()
                .all(|&v| v == 5.0),
            "the fetched tensor must share the var's storage"
        );
        assert!(
            derived
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .flatten()
                .all(|&v| v == 1.0),
            "a derived copy must NOT see the update (it owns fresh storage)"
        );
    }

    #[test]
    fn excluded_names_pass_through_raw_without_minting_vars() {
        fn exclude_experts(name: &str) -> bool {
            name.contains(".experts.")
        }
        let (vb, reg) = registry_varbuilder(
            map(&["mlp.experts.0.weight", "mlp.gate.weight"]),
            DType::F32,
            &dev(),
            exclude_experts,
        );
        vb.pp("mlp.experts.0").get((2, 3), "weight").unwrap();
        vb.pp("mlp.gate").get((2, 3), "weight").unwrap();
        assert_eq!(
            reg.names().unwrap(),
            vec!["mlp.gate.weight"],
            "the excluded fetch must not register; the plain one must"
        );
    }

    #[test]
    fn explicit_register_returns_a_storage_sharing_handle() {
        let reg = VarRegistry::new();
        let packed = Tensor::full(2.0f32, (4, 2, 3), &dev()).unwrap();
        let handle = reg
            .register("layers.0.mlp.experts.gate_up_packed", &packed)
            .unwrap();
        let next = Tensor::full(7.0f32, (4, 2, 3), &dev()).unwrap();
        reg.vars().unwrap()[0].set(&next).unwrap();
        let got = handle.to_vec3::<f32>().unwrap();
        assert!(
            got.iter().flatten().flatten().all(|&v| v == 7.0),
            "the registered handle must share the var's storage"
        );
        // And the SOURCE tensor stays untouched — from_tensor copied it.
        let src = packed.to_vec3::<f32>().unwrap();
        assert!(src.iter().flatten().flatten().all(|&v| v == 2.0));
    }
}
