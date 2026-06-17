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
}

/// Read the model's `eos_token_id` from `<dir>/config.json`.
///
/// Looks under the multimodal wrapper's `text_config` first (the Qwen3.5/3.6
/// checkpoint shape), then the top level — so both a flat text-only config and a
/// wrapped one resolve. Returns `Ok(None)` when the field is absent (a base model
/// with no EOS) or is not a single non-negative integer that fits a `u32`. A
/// **list-valued** `eos_token_id` (some chat models declare several stop ids) is
/// intentionally treated as `None`:
/// [`GenConfig::eos_token_id`](crate::policy::GenConfig::eos_token_id) carries a
/// single id, so a multi-EOS model must pick one explicitly rather than have one
/// silently chosen here.
///
/// # Errors
///
/// Returns [`HfError::Io`] if `<dir>/config.json` cannot be read, or
/// [`HfError::Parse`] if it is not valid JSON.
pub fn eos_from_config<P: AsRef<Path>>(dir: P) -> Result<Option<u32>, HfError> {
    let path = dir.as_ref().join("config.json");
    let bytes = fs::read(&path).map_err(|source| HfError::Io {
        path: path.clone(),
        source,
    })?;
    let json: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|source| HfError::Parse { path, source })?;
    Ok(eos_from_json(&json))
}

/// Extract a single `u32` `eos_token_id` from a parsed config value (the pure core
/// of [`eos_from_config`]). Checks `text_config.eos_token_id` (the multimodal
/// wrapper shape) before the top-level field. `None` unless the resolved value is a
/// single integer in `u32` range.
fn eos_from_json(json: &serde_json::Value) -> Option<u32> {
    json.pointer("/text_config/eos_token_id")
        .or_else(|| json.get("eos_token_id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
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

    #[test]
    fn eos_from_json_reads_a_single_integer() {
        let json = serde_json::json!({ "eos_token_id": 151_645, "hidden_size": 16 });
        assert_eq!(eos_from_json(&json), Some(151_645));
    }

    #[test]
    fn eos_from_json_is_none_when_absent() {
        let json = serde_json::json!({ "hidden_size": 16 });
        assert_eq!(eos_from_json(&json), None);
    }

    #[test]
    fn eos_from_json_prefers_the_nested_text_config() {
        // The Qwen3.5/3.6 multimodal-wrapper shape: EOS nested under text_config,
        // taken ahead of any top-level value.
        let json = serde_json::json!({
            "eos_token_id": 1,
            "text_config": { "eos_token_id": 151_645 },
        });
        assert_eq!(eos_from_json(&json), Some(151_645));
    }

    #[test]
    fn eos_from_json_is_none_for_a_multi_eos_list() {
        // A list-valued eos_token_id is deliberately not auto-picked.
        let json = serde_json::json!({ "eos_token_id": [151_645, 151_643] });
        assert_eq!(eos_from_json(&json), None);
    }

    #[test]
    fn eos_from_json_is_none_for_an_out_of_range_or_wrong_type() {
        assert_eq!(
            eos_from_json(&serde_json::json!({ "eos_token_id": -1 })),
            None
        );
        assert_eq!(
            eos_from_json(&serde_json::json!({ "eos_token_id": u64::from(u32::MAX) + 1 })),
            None
        );
        assert_eq!(
            eos_from_json(&serde_json::json!({ "eos_token_id": "151645" })),
            None
        );
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
