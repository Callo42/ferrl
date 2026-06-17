//! CUDA driver-compatibility preflight.
//!
//! The CUDA kernels `ferrl` runs (candle's GPU backend, `candle-kernels`) are
//! compiled to **PTX** at *build* time by whatever CUDA toolkit (`nvcc`) the build
//! machine has. At *run* time the NVIDIA **driver** JIT-compiles that PTX for the
//! GPU, and it rejects any PTX whose ISA version is newer than the driver supports
//! — surfacing the cryptic `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (CUDA driver error
//! `222`) at the **first** GPU kernel load. `CUDA_COMPUTE_CAP` sets the GPU SM
//! *architecture*, not the PTX ISA; only the toolkit version sets the ISA.
//!
//! This module turns that failure mode into something actionable:
//!
//! - [`translate_ptx_error`] rewrites a `222` (matched on the typed cudarc error
//!   code, recursing through candle's error-wrapper variants) into a message that
//!   says how to fix it — rebuild with an older toolkit, or upgrade the driver.
//!   `ferrl`'s own GPU forward applies it automatically, and it is re-exported for
//!   callers who wrap their own load paths.
//! - [`guard_first_kernel`] forces the first kernel JIT at a clean chokepoint so
//!   the (translated) failure surfaces early rather than buried in a forward.
//! - [`check_driver_compat`] is an optional, **warn-only** proactive check: it
//!   compares the driver's reported CUDA version against the PTX ISA this binary
//!   was actually built with (read from the embedded kernel PTX) and reports
//!   [`CompatReport::TooOld`] *before* the cryptic error can fire. It never blocks.
//!
//! Without the `cuda` feature every entry point is a no-op (there is no GPU build
//! to be incompatible with), so callers need no `cfg`.
//!
//! The compatibility numbers (CUDA toolkit ↔ PTX ISA ↔ minimum Linux driver) mirror
//! the table in the project README's "GPU builds" section.

// The pure, CUDA-type-free compatibility logic: PTX-version parsing, the
// CUDA↔ISA↔driver tables, and message rendering. Gated on `any(cuda, test)` so the
// default build's test pass exercises it (no GPU needed) without leaving dead code
// in the default non-test build.
#[cfg(any(feature = "cuda", test))]
pub(crate) mod logic {
    /// The stable variant-name token cudarc's `DriverError` `Debug` renders for the
    /// driver error code `222`. Used only as a string co-signal fallback; the typed
    /// path matches the numeric code directly.
    pub(crate) const PTX_TOKEN: &str = "CUDA_ERROR_UNSUPPORTED_PTX_VERSION";

    /// A `(major, minor)` CUDA / PTX-ISA version pair.
    pub(crate) type Version = (u32, u32);

    /// The optional driver facts an enriched reactive message names:
    /// `(driver_cuda, driver_max_isa, min_driver_for_this_build)`.
    pub(crate) type DriverFacts<'a> = (Version, Version, &'a str);

    /// CUDA toolkit version → the maximum PTX ISA version it emits (Linux `x86_64`).
    /// Sourced from NVIDIA's PTX ISA release notes. Ascending; `12.6` reuses ISA `8.5`,
    /// then `12.7` (the r565 generation; a 565 driver reports CUDA `12.7`) introduces
    /// ISA `8.6`, `12.8` → `8.7`, `12.9` → `8.8`.
    pub(crate) const CUDA_TO_MAX_ISA: &[(Version, Version)] = &[
        ((11, 8), (7, 8)),
        ((12, 0), (8, 0)),
        ((12, 1), (8, 1)),
        ((12, 2), (8, 2)),
        ((12, 3), (8, 3)),
        ((12, 4), (8, 4)),
        ((12, 5), (8, 5)),
        ((12, 6), (8, 5)),
        ((12, 7), (8, 6)),
        ((12, 8), (8, 7)),
        ((12, 9), (8, 8)),
    ];

    /// PTX ISA version → the minimum Linux `x86_64` NVIDIA driver that can JIT it (the
    /// driver that first introduced JIT support for that ISA). This is the floor error
    /// `222` keys on: it depends on the embedded PTX `.version`, NOT the toolkit. So a
    /// CUDA `12.6` build (which emits ISA `8.5`) needs `555.42.02` here — even though the
    /// README's per-toolkit column lists `12.6` against `560.28.03` (NVIDIA's minimum for
    /// the full `12.6` toolkit runtime, a stricter but unrelated requirement). Both are
    /// correct for their purpose. Sourced from NVIDIA's CUDA Toolkit release notes.
    pub(crate) const ISA_TO_MIN_DRIVER: &[(Version, &str)] = &[
        ((7, 8), "520.61.05"),
        ((8, 0), "525.60.13"),
        ((8, 1), "530.30.02"),
        ((8, 2), "535.54.03"),
        ((8, 3), "545.23.06"),
        ((8, 4), "550.54.14"),
        ((8, 5), "555.42.02"),
        ((8, 6), "565.57.01"),
        ((8, 7), "570.26"),
        ((8, 8), "575.51.03"),
    ];

    /// Whether `s` looks like the PTX-version driver error. Requires the variant
    /// token **and** a co-token, so an unrelated message that merely mentions the
    /// token cannot be mis-flagged.
    pub(crate) fn looks_like_unsupported_ptx(s: &str) -> bool {
        s.contains(PTX_TOKEN) && (s.contains("DriverError") || s.contains("222"))
    }

    /// Parse the first `.version M.m` directive of a PTX module (every PTX module
    /// begins with one). `None` if no well-formed directive is found.
    pub(crate) fn parse_ptx_version(ptx: &str) -> Option<(u32, u32)> {
        for line in ptx.lines() {
            if let Some(rest) = line.trim().strip_prefix(".version") {
                let (maj, min) = rest.trim().split_once('.')?;
                return Some((maj.trim().parse().ok()?, min.trim().parse().ok()?));
            }
        }
        None
    }

    /// Decode `cudaDriverGetVersion`'s packed integer (`1000*major + 10*minor`),
    /// e.g. `12080` → `(12, 8)`. `None` for a non-positive (unavailable) version.
    pub(crate) fn decode_cuda_version(v: i32) -> Option<(u32, u32)> {
        if v <= 0 {
            return None;
        }
        let v = v as u32;
        Some((v / 1000, (v % 1000) / 10))
    }

    /// The maximum PTX ISA a driver reporting `driver_cuda` can JIT. Returns `None`
    /// when `driver_cuda` is newer than the highest tabulated row — so a driver newer
    /// than this table is treated as "unknown", never as too old.
    pub(crate) fn driver_max_isa(driver_cuda: (u32, u32)) -> Option<(u32, u32)> {
        match CUDA_TO_MAX_ISA.last() {
            Some(&(top_cuda, _)) if driver_cuda > top_cuda => None,
            _ => CUDA_TO_MAX_ISA
                .iter()
                .filter(|&&(cuda, _)| cuda <= driver_cuda)
                .map(|&(_, isa)| isa)
                .next_back(),
        }
    }

    /// The minimum Linux `x86_64` driver string for PTX ISA `isa` (exact row, else the
    /// next-higher tabulated ISA — a conservative over-estimate).
    pub(crate) fn min_driver_for_isa(isa: (u32, u32)) -> Option<&'static str> {
        ISA_TO_MIN_DRIVER
            .iter()
            .find(|&&(row, _)| row >= isa)
            .map(|&(_, drv)| drv)
    }

    /// The newest CUDA toolkit whose emitted PTX ISA is `<= isa`.
    pub(crate) fn max_cuda_for_isa(isa: (u32, u32)) -> Option<(u32, u32)> {
        CUDA_TO_MAX_ISA
            .iter()
            .filter(|&&(_, row)| row <= isa)
            .map(|&(cuda, _)| cuda)
            .next_back()
    }

    /// The actionable reactive message for a `222`. When `driver` is
    /// `Some((driver_cuda, driver_max_isa, min_driver_for_this_build))` the message
    /// names the driver's CUDA ceiling and the exact driver to upgrade to; otherwise
    /// it points the user at `nvidia-smi` and the README table.
    pub(crate) fn unsupported_ptx_message(driver: Option<DriverFacts<'_>>) -> String {
        let mut s = String::from(
            "ferrl: CUDA driver too old for this build (CUDA_ERROR_UNSUPPORTED_PTX_VERSION, \
             CUDA driver error 222).\n\n",
        );
        if let Some(((dmaj, dmin), (imaj, imin), min_driver)) = driver {
            s.push_str(&format!(
                "The first GPU kernel load failed: this ferrl binary's CUDA kernels were compiled \
                 to a PTX ISA newer than your NVIDIA driver can run. Your driver supports up to \
                 CUDA {dmaj}.{dmin} (PTX ISA {imaj}.{imin}); it JIT-compiles PTX at first kernel \
                 load and rejects anything newer.\n\n\
                 Fix it either way:\n  \
                 1. Rebuild ferrl with a CUDA toolkit your driver supports (toolkit version <= \
                 CUDA {dmaj}.{dmin}); or\n  \
                 2. Upgrade your NVIDIA driver to at least {min_driver} (Linux x86_64; Windows \
                 minimums differ).\n\n",
            ));
        } else {
            s.push_str(
                "The first GPU kernel load failed: this ferrl binary's CUDA kernels were compiled \
                 to a PTX ISA newer than your NVIDIA driver can run. The driver JIT-compiles PTX at \
                 first kernel load and rejects anything newer than it supports.\n\n\
                 Fix it either way:\n  \
                 1. Rebuild ferrl with an older CUDA toolkit (one whose PTX ISA your driver \
                 accepts); or\n  \
                 2. Upgrade your NVIDIA driver to a version new enough for the toolkit ferrl was \
                 built with.\n\n",
            );
        }
        s.push_str(
            "Run `nvidia-smi` to see your driver; its top-right \"CUDA Version\" is your driver's \
             maximum supported CUDA. Note: CUDA_COMPUTE_CAP sets the GPU SM architecture, NOT the \
             PTX ISA \u{2014} lowering it will NOT fix this. See the project README, \"GPU builds\" \
             \u{2192} \"CUDA driver compatibility\", for the full toolkit/driver table.",
        );
        s
    }

    /// The proactive early-warning message. The driver→ISA mapping is a heuristic
    /// (hence "typically"); the reactive guard is the authoritative check.
    pub(crate) fn proactive_warn_message(
        built_isa: (u32, u32),
        driver_cuda: (u32, u32),
        driver_max_isa: (u32, u32),
        min_driver: &str,
    ) -> String {
        let (bmaj, bmin) = built_isa;
        let (dmaj, dmin) = driver_cuda;
        let (imaj, imin) = driver_max_isa;
        format!(
            "ferrl preflight: your NVIDIA driver reports CUDA {dmaj}.{dmin} (typically max PTX ISA \
             {imaj}.{imin}), but this binary's kernels target PTX ISA {bmaj}.{bmin}. The first GPU \
             kernel will likely fail with CUDA_ERROR_UNSUPPORTED_PTX_VERSION. Rebuild ferrl with a \
             CUDA toolkit <= {dmaj}.{dmin}, or upgrade your NVIDIA driver to >= {min_driver} (see \
             README \"GPU builds\").",
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_ptx_version_reads_the_directive() {
            assert_eq!(
                parse_ptx_version(".version 8.7\n.target sm_80\n"),
                Some((8, 7))
            );
            // Leading blank/comment lines before `.version` are tolerated.
            assert_eq!(
                parse_ptx_version("//\n\n  .version 7.8\n.target sm_70\n"),
                Some((7, 8))
            );
            // No directive, or a malformed one, yields None.
            assert_eq!(parse_ptx_version(".target sm_80\n"), None);
            assert_eq!(parse_ptx_version(".version eight\n"), None);
            assert_eq!(parse_ptx_version(""), None);
        }

        #[test]
        fn decode_cuda_version_unpacks_the_int() {
            assert_eq!(decode_cuda_version(12080), Some((12, 8)));
            assert_eq!(decode_cuda_version(11080), Some((11, 8)));
            assert_eq!(decode_cuda_version(12040), Some((12, 4)));
            assert_eq!(decode_cuda_version(0), None);
            assert_eq!(decode_cuda_version(-5), None);
        }

        #[test]
        fn driver_max_isa_clamps_down_and_never_overshoots_a_newer_driver() {
            // Exact rows, including the r565 generation (a 565 driver reports CUDA 12.7).
            assert_eq!(driver_max_isa((12, 8)), Some((8, 7)));
            assert_eq!(driver_max_isa((12, 7)), Some((8, 6)));
            // A driver CUDA with no exact row clamps DOWN to the nearest lower tabulated
            // toolkit (11.9 -> 11.8 -> ISA 7.8), never UP.
            assert_eq!(driver_max_isa((11, 9)), Some((7, 8)));
            // A driver NEWER than the whole table -> None, so the caller never emits
            // a false "too old" for a perfectly capable new driver.
            assert_eq!(driver_max_isa((13, 0)), None);
            assert_eq!(driver_max_isa((99, 9)), None);
        }

        #[test]
        fn min_driver_for_isa_lookups() {
            assert_eq!(min_driver_for_isa((7, 8)), Some("520.61.05"));
            assert_eq!(min_driver_for_isa((8, 4)), Some("550.54.14"));
            assert_eq!(min_driver_for_isa((8, 6)), Some("565.57.01"));
            assert_eq!(min_driver_for_isa((8, 7)), Some("570.26"));
        }

        #[test]
        fn max_cuda_for_isa_lookups() {
            // Max toolkit for a given ISA: 8.5 is reused by 12.6 (the newest), and 8.6
            // is the r565 generation (CUDA 12.7).
            assert_eq!(max_cuda_for_isa((8, 4)), Some((12, 4)));
            assert_eq!(max_cuda_for_isa((8, 5)), Some((12, 6)));
            assert_eq!(max_cuda_for_isa((8, 6)), Some((12, 7)));
            assert_eq!(max_cuda_for_isa((8, 7)), Some((12, 8)));
        }

        #[test]
        fn looks_like_unsupported_ptx_needs_a_co_token() {
            assert!(looks_like_unsupported_ptx(
                "DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION, ...)"
            ));
            assert!(looks_like_unsupported_ptx(
                "error 222: CUDA_ERROR_UNSUPPORTED_PTX_VERSION"
            ));
            // The token alone (no DriverError / 222 co-token) does NOT match — no
            // false positive on an unrelated log line that merely names the constant.
            assert!(!looks_like_unsupported_ptx(
                "see CUDA_ERROR_UNSUPPORTED_PTX_VERSION in the docs"
            ));
            assert!(!looks_like_unsupported_ptx("shape mismatch"));
        }

        #[test]
        fn reactive_message_known_driver_is_actionable() {
            let m = unsupported_ptx_message(Some(((12, 4), (8, 4), "570.26")));
            assert!(m.contains("CUDA driver error 222"));
            assert!(m.contains("toolkit version <= CUDA 12.4"));
            assert!(m.contains("at least 570.26"));
            assert!(m.contains("CUDA_COMPUTE_CAP sets the GPU SM architecture"));
            assert!(m.contains("nvidia-smi"));
        }

        #[test]
        fn reactive_message_unknown_driver_still_actionable() {
            let m = unsupported_ptx_message(None);
            assert!(m.contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION"));
            assert!(m.contains("older CUDA toolkit"));
            assert!(m.contains("CUDA_COMPUTE_CAP"));
            // The unknown form omits the numeric ceiling clauses.
            assert!(!m.contains("supports up to CUDA"));
        }

        #[test]
        fn proactive_message_hedges_the_heuristic() {
            let m = proactive_warn_message((8, 7), (12, 4), (8, 4), "570.26");
            assert!(m.contains("reports CUDA 12.4"));
            assert!(m.contains("target PTX ISA 8.7"));
            assert!(m.contains("typically max PTX ISA 8.4"));
            assert!(m.contains(">= 570.26"));
        }

        #[test]
        fn no_repo_message_leaks_cluster_specifics() {
            // Governance regression guard: nothing destined for a user's terminal may
            // carry private cluster paths / node names / usernames.
            let messages = [
                unsupported_ptx_message(Some(((12, 4), (8, 4), "570.26"))),
                unsupported_ptx_message(None),
                proactive_warn_message((8, 7), (12, 4), (8, 4), "570.26"),
            ];
            for m in &messages {
                // Broad deny-list: substrings that would catch any cluster path, node, or
                // Slurm token if a future edit ever interpolated one into a user message.
                for forbidden in [
                    "/home/",
                    "/tmp/ferrl-",
                    "n00",
                    "srun",
                    "sbatch",
                    "--partition",
                    "--gres",
                ] {
                    assert!(!m.contains(forbidden), "message leaks {forbidden:?}: {m}");
                }
            }
        }
    }
}

/// A snapshot of how this binary's built PTX ISA compares to the runtime driver.
///
/// Produced by [`check_driver_compat`]. The version tuples are `(major, minor)`;
/// `min_driver` is a Linux `x86_64` NVIDIA driver string. [`CompatReport::Unknown`]
/// (with a reason) is returned whenever a needed fact could not be read — including
/// on the non-`cuda` build — and is deliberately silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatReport {
    /// The driver can run this build's PTX.
    Ok {
        /// The PTX ISA this binary's kernels were built to.
        built_isa: (u32, u32),
        /// The driver's reported maximum CUDA version.
        driver_cuda: (u32, u32),
        /// The maximum PTX ISA the driver can JIT.
        driver_max_isa: (u32, u32),
    },
    /// The driver looks too old for this build's PTX — the first kernel will likely
    /// fail with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Warn-only; this is a heuristic.
    TooOld {
        /// The PTX ISA this binary's kernels were built to.
        built_isa: (u32, u32),
        /// The driver's reported maximum CUDA version.
        driver_cuda: (u32, u32),
        /// The maximum PTX ISA the driver can JIT.
        driver_max_isa: (u32, u32),
        /// The newest CUDA toolkit a user could rebuild with for this driver.
        max_cuda: (u32, u32),
        /// The minimum Linux driver that would run this build.
        min_driver: &'static str,
    },
    /// A needed fact was unavailable (named here); no judgement is made.
    Unknown(&'static str),
}

impl CompatReport {
    /// A user-facing warning string for an apparent mismatch ([`CompatReport::TooOld`]),
    /// else `None`. Callers typically `tracing::warn!` it.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn warning(&self) -> Option<String> {
        match *self {
            CompatReport::TooOld {
                built_isa,
                driver_cuda,
                driver_max_isa,
                min_driver,
                ..
            } => Some(logic::proactive_warn_message(
                built_isa,
                driver_cuda,
                driver_max_isa,
                min_driver,
            )),
            _ => None,
        }
    }

    /// Always `None` without the `cuda` feature — there is no GPU build to warn about.
    #[cfg(not(feature = "cuda"))]
    #[must_use]
    pub fn warning(&self) -> Option<String> {
        None
    }
}

/// Translate a `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (driver error `222`) into an
/// actionable rebuild/upgrade message, recursing through candle's error-wrapper
/// variants to find it; any other error is returned unchanged.
///
/// `ferrl`'s GPU forward applies this automatically, so most callers never need it
/// directly; it is exposed for code that drives its own CUDA load paths. Without the
/// `cuda` feature it is the identity function.
#[cfg(feature = "cuda")]
#[must_use]
pub fn translate_ptx_error(err: candle_core::Error) -> candle_core::Error {
    if cuda_only::is_unsupported_ptx(&err) {
        candle_core::Error::Msg(logic::unsupported_ptx_message(cuda_only::driver_hint()))
    } else {
        err
    }
}

/// Without the `cuda` feature there is no CUDA error to translate; returns `err`.
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn translate_ptx_error(err: candle_core::Error) -> candle_core::Error {
    err
}

/// Force the first GPU kernel to JIT now (a tiny throwaway op), so a driver-too-old
/// PTX mismatch surfaces as the actionable [`translate_ptx_error`] message at this
/// clean chokepoint instead of buried in the first real forward.
///
/// # Errors
///
/// Returns the translated actionable error if the driver rejects this build's PTX
/// (or any other error the probe op raises).
#[cfg(feature = "cuda")]
pub fn guard_first_kernel(device: &candle_core::Device) -> candle_core::Result<()> {
    let probe = candle_core::Tensor::ones((1usize, 1usize), candle_core::DType::F32, device)
        .and_then(|t| t.affine(1.0, 0.0));
    probe.map(|_| ()).map_err(translate_ptx_error)
}

/// Without the `cuda` feature there are no GPU kernels to load; a no-op `Ok(())`.
///
/// # Errors
///
/// Never returns an error on the non-`cuda` build.
#[cfg(not(feature = "cuda"))]
pub fn guard_first_kernel(_device: &candle_core::Device) -> candle_core::Result<()> {
    Ok(())
}

/// Proactively compare the driver's reported CUDA version against the PTX ISA this
/// binary was built with (read from the embedded kernel PTX — ground truth, not a
/// guess), returning a [`CompatReport`]. **Never errors and never blocks**: any
/// unreadable input, or a driver newer than the built-in table, yields
/// [`CompatReport::Unknown`]. The authoritative check is [`guard_first_kernel`].
///
/// The built ISA can be overridden for exotic split-toolchain builds by setting
/// `FERRL_BUILT_CUDA="MAJOR.MINOR"`; this only affects the warning, never the guard.
#[cfg(feature = "cuda")]
#[must_use]
pub fn check_driver_compat(device: &candle_core::Device) -> CompatReport {
    if device.as_cuda_device().is_err() {
        return CompatReport::Unknown("not a CUDA device");
    }
    let Some(built_isa) = cuda_only::built_ptx_isa() else {
        return CompatReport::Unknown("could not read the built PTX ISA");
    };
    let Some(driver_cuda) = cuda_only::driver_cuda_version() else {
        return CompatReport::Unknown("driver version query unavailable");
    };
    let Some(driver_max_isa) = logic::driver_max_isa(driver_cuda) else {
        return CompatReport::Unknown("driver newer than the compatibility table");
    };
    if built_isa > driver_max_isa {
        CompatReport::TooOld {
            built_isa,
            driver_cuda,
            driver_max_isa,
            max_cuda: logic::max_cuda_for_isa(driver_max_isa).unwrap_or(driver_cuda),
            min_driver: logic::min_driver_for_isa(built_isa).unwrap_or("a newer driver"),
        }
    } else {
        CompatReport::Ok {
            built_isa,
            driver_cuda,
            driver_max_isa,
        }
    }
}

/// Always [`CompatReport::Unknown`] without the `cuda` feature.
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn check_driver_compat(_device: &candle_core::Device) -> CompatReport {
    CompatReport::Unknown("built without the cuda feature")
}

// The CUDA-typed internals: error classification against the typed cudarc code, the
// embedded-PTX read, and the safe driver-version query. Only compiled with `cuda`.
#[cfg(feature = "cuda")]
mod cuda_only {
    use super::logic;
    use candle_core::Error;

    /// Recurse through candle's error-wrapper variants to decide whether the root
    /// cause is `DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION)`.
    pub(super) fn is_unsupported_ptx(err: &Error) -> bool {
        match err {
            Error::WithBacktrace { inner, .. }
            | Error::WithPath { inner, .. }
            | Error::Context { inner, .. } => is_unsupported_ptx(inner),
            Error::WrappedContext { wrapped, .. } => {
                wrapped
                    .downcast_ref::<Error>()
                    .is_some_and(is_unsupported_ptx)
                    || wrapped
                        .downcast_ref::<candle_core::cuda::CudaError>()
                        .is_some_and(cuda_err_is_ptx)
            }
            Error::Cuda(boxed) => boxed
                .downcast_ref::<candle_core::cuda::CudaError>()
                .is_some_and(cuda_err_is_ptx),
            Error::Wrapped(_) => logic::looks_like_unsupported_ptx(&err.to_string()),
            _ => false,
        }
    }

    /// Match the typed cudarc error code `222` inside candle's `CudaError`.
    fn cuda_err_is_ptx(ce: &candle_core::cuda::CudaError) -> bool {
        use candle_core::cuda::cudarc::driver::sys::cudaError_enum::CUDA_ERROR_UNSUPPORTED_PTX_VERSION as PTX222;
        use candle_core::cuda::CudaError;
        match ce {
            CudaError::Cuda(d) => d.0 == PTX222,
            CudaError::Load { cuda, .. } => cuda.0 == PTX222,
            _ => false,
        }
    }

    /// Parse `MAJOR.MINOR` and map it through the toolkit→ISA table.
    fn env_override_isa() -> Option<(u32, u32)> {
        let v = std::env::var("FERRL_BUILT_CUDA").ok()?;
        let (maj, min) = v.split_once('.')?;
        let cuda = (maj.trim().parse().ok()?, min.trim().parse().ok()?);
        logic::driver_max_isa(cuda)
    }

    /// The PTX ISA this binary's `candle-kernels` were compiled to — ground truth
    /// from the embedded PTX text, or the `FERRL_BUILT_CUDA` override.
    pub(super) fn built_ptx_isa() -> Option<(u32, u32)> {
        env_override_isa()
            .or_else(|| logic::parse_ptx_version(candle_core::cuda::kernels::AFFINE.ptx()))
    }

    /// The driver's reported CUDA version, via cudarc's safe `get_driver_version`.
    pub(super) fn driver_cuda_version() -> Option<(u32, u32)> {
        cudarc::runtime::result::version::get_driver_version()
            .ok()
            .and_then(logic::decode_cuda_version)
    }

    /// `(driver_cuda, driver_max_isa, min_driver_for_this_build)` for an enriched
    /// reactive message — `None` unless both the build ISA and the driver are readable.
    pub(super) fn driver_hint() -> Option<logic::DriverFacts<'static>> {
        let built = built_ptx_isa()?;
        let driver_cuda = driver_cuda_version()?;
        let driver_max = logic::driver_max_isa(driver_cuda)?;
        let min_driver = logic::min_driver_for_isa(built)?;
        Some((driver_cuda, driver_max, min_driver))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use candle_core::cuda::cudarc::driver::sys::cudaError_enum;
        use candle_core::cuda::cudarc::driver::DriverError;
        use candle_core::cuda::CudaError;

        /// A leaf candle error carrying the typed `222` driver code.
        fn ptx_leaf() -> Error {
            Error::Cuda(Box::new(CudaError::Cuda(DriverError(
                cudaError_enum::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
            ))))
        }

        #[test]
        fn typed_222_is_detected_and_translated() {
            assert!(is_unsupported_ptx(&ptx_leaf()));
            let translated = super::super::translate_ptx_error(ptx_leaf());
            assert!(translated.to_string().contains(logic::PTX_TOKEN));
            assert!(translated.to_string().contains("error 222"));
        }

        #[test]
        fn non_222_and_plain_errors_pass_through_unchanged() {
            let oom = Error::Cuda(Box::new(CudaError::Cuda(DriverError(
                cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY,
            ))));
            assert!(!is_unsupported_ptx(&oom));
            assert!(!is_unsupported_ptx(&Error::Msg("a shape mismatch".into())));
            // translate is a no-op on a non-PTX error.
            let passed = super::super::translate_ptx_error(Error::Msg("keep me".into()));
            assert!(passed.to_string().contains("keep me"));
        }

        #[test]
        fn recursion_finds_222_through_every_wrapper() {
            // candle's Error is not #[non_exhaustive]; build each wrapper directly so
            // the test is deterministic regardless of RUST_BACKTRACE. The translator
            // must see through every layer the runtime can produce.
            let wrappers: Vec<Error> = vec![
                Error::WithBacktrace {
                    inner: Box::new(ptx_leaf()),
                    backtrace: Box::new(std::backtrace::Backtrace::capture()),
                },
                Error::WithPath {
                    inner: Box::new(ptx_leaf()),
                    path: std::path::PathBuf::from("/tmp/x"),
                },
                Error::Context {
                    inner: Box::new(ptx_leaf()),
                    context: Box::new("loading weights"),
                },
                Error::WrappedContext {
                    wrapped: Box::new(ptx_leaf()),
                    context: "loading weights".to_string(),
                },
                // Nested: context(with_path(leaf)).
                Error::Context {
                    inner: Box::new(Error::WithPath {
                        inner: Box::new(ptx_leaf()),
                        path: std::path::PathBuf::from("/tmp/y"),
                    }),
                    context: Box::new("outer"),
                },
            ];
            for w in &wrappers {
                assert!(is_unsupported_ptx(w), "missed the 222 through {w}");
            }
            // A wrapper around a NON-222 leaf must not match.
            let wrapped_oom = Error::WithPath {
                inner: Box::new(Error::Msg("not a ptx error".into())),
                path: std::path::PathBuf::from("/tmp/z"),
            };
            assert!(!is_unsupported_ptx(&wrapped_oom));
        }

        #[test]
        fn driver_error_debug_renders_the_stable_token() {
            // The classifier's string fallback and the format-stability of the typed
            // path both rest on cudarc's DriverError Debug rendering the variant name.
            // Pin it so a cudarc bump that changed the rendering is caught.
            let e = DriverError(cudaError_enum::CUDA_ERROR_UNSUPPORTED_PTX_VERSION);
            assert!(format!("{e:?}").contains(logic::PTX_TOKEN));
        }
    }
}

// The always-present public API in its non-`cuda` (no-op) form: callers need no `cfg`,
// so the inert behavior is pinned even on the default build.
#[cfg(all(test, not(feature = "cuda")))]
mod cpu_noop_tests {
    use super::*;
    use candle_core::{Device, Error};

    #[test]
    fn the_preflight_is_inert_without_the_cuda_feature() {
        let dev = Device::Cpu;
        // The reactive guard is a no-op Ok, the proactive check is silent Unknown,
        // and there is nothing to warn about.
        assert!(guard_first_kernel(&dev).is_ok());
        assert!(matches!(
            check_driver_compat(&dev),
            CompatReport::Unknown(_)
        ));
        assert!(check_driver_compat(&dev).warning().is_none());
        // translate is the identity: the error passes through untouched.
        let passed = translate_ptx_error(Error::Msg("untouched".into()));
        assert!(passed.to_string().contains("untouched"));
    }
}
