//! A real-model tokenizer adapter.
//!
//! [`HfTokenizer`] wraps a Hugging Face `tokenizers::Tokenizer` (loaded from a
//! `tokenizer.json`) and implements the trainer's [`TokenizerLike`] bridge, so a
//! real model (e.g. `Qwen3-0.6B-Base`) plugs into the GRPO loop the same way the
//! P2 toy's char codec does. The toy keeps a trivial in-test codec; this is the
//! production path that loads an actual model's vocabulary.
//!
//! ## Why the trait stays infallible
//!
//! [`TokenizerLike::encode`] / [`decode`](TokenizerLike::decode) are *total* by
//! contract — the trainer calls them in the hot rollout loop, and neither the
//! [`Policy`](crate::policy::Policy) nor the [`RewardFn`](crate::reward::RewardFn)
//! should have to thread a tokenizer error. Construction
//! ([`HfTokenizer::from_file`]) is therefore where loading/validation failures
//! surface; at call time the wrapper is total, decoding lossily rather than
//! erroring (the documented [`TokenizerLike`] behavior). In practice a fast
//! tokenizer does not fail to encode valid UTF-8 or to decode in-vocab ids.
//!
//! ## Special tokens
//!
//! For a *base* model under GRPO the prompt is encoded **without** added special
//! tokens: the caller owns any chat/template framing, and the trainer itself
//! concatenates the prompt ids with the sampled completion ids. Decoding **skips**
//! special tokens so the [`RewardFn`](crate::reward::RewardFn) scores clean text.

use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

use crate::trainer::TokenizerLike;

/// Errors raised while constructing a [`HfTokenizer`].
///
/// Construction is the one fallible point in this adapter — the
/// [`TokenizerLike`] call path is total by contract (see the module docs) — so a
/// single load variant covers the whole surface. This is a typed library error
/// rather than a boxed `anyhow` value: a downstream caller can match on it, and
/// the crate's public API stays `anyhow`-free (`anyhow` is reserved for examples
/// and glue code).
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// The file at `path` could not be read, or did not contain a valid
    /// `tokenizers` fast-tokenizer definition.
    #[error("failed to load tokenizer from {path:?}: {source}")]
    Load {
        /// The `tokenizer.json` path that failed to load.
        path: PathBuf,
        /// The underlying error reported by the `tokenizers` crate (a boxed
        /// trait object; `tokenizers::Error` is not a concrete type we can name).
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// A [`TokenizerLike`] backed by a Hugging Face fast tokenizer.
///
/// Construct it once with [`HfTokenizer::from_file`] (e.g. over a model's
/// `tokenizer.json`); the resulting value is the production bridge handed to the
/// [`Trainer`](crate::trainer::Trainer) in place of the toy codec.
#[derive(Debug)]
pub struct HfTokenizer {
    inner: Tokenizer,
}

impl HfTokenizer {
    /// Load a fast tokenizer from a `tokenizer.json` file.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::Load`] if `path` cannot be read or does not
    /// contain a valid `tokenizers` definition.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        // `tokenizers::from_file` returns a boxed error (`tokenizers::Error`);
        // capture it as the typed variant's `source` so the chain is preserved.
        let inner = Tokenizer::from_file(path).map_err(|source| TokenizerError::Load {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self { inner })
    }
}

impl TokenizerLike for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        // Base-model prompts: no added special tokens (see module docs). A real
        // fast tokenizer does not fail on valid UTF-8; honor the infallible
        // `TokenizerLike` contract with an empty fallback, but log the (genuine)
        // failure so it cannot masquerade as an empty prompt in the rollout loop.
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                tracing::error!("tokenizer encode failed, returning empty ids: {e}");
                Vec::new()
            }
        }
    }

    fn decode(&self, ids: &[u32]) -> String {
        // Skip special tokens so the reward scores clean text; decode lossily
        // (documented `TokenizerLike` behavior), but log a genuine decode error
        // rather than silently returning an empty completion.
        match self.inner.decode(ids, true) {
            Ok(text) => text,
            Err(e) => {
                tracing::error!("tokenizer decode failed, returning empty string: {e}");
                String::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the committed tiny `WordLevel` tokenizer fixture (vocab of a few
    /// known words). Loadable offline in CI — the real model tokenizer is only
    /// reachable from the `#[ignore]`d real-weights tests.
    fn fixture() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tiny_tokenizer.json"
        )
        .to_string()
    }

    #[test]
    fn round_trips_through_the_fixture() {
        let tok = HfTokenizer::from_file(fixture()).unwrap();
        let ids = tok.encode("hello world");
        // "hello" -> 0, "world" -> 1 in the fixture vocab.
        assert_eq!(ids, vec![0, 1]);
        let text = tok.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn unknown_words_map_to_unk_and_decode_back() {
        let tok = HfTokenizer::from_file(fixture()).unwrap();
        // "absent" is out of vocab -> the [UNK] id (2).
        assert_eq!(tok.encode("absent"), vec![2]);
        // [UNK] is an ordinary (non-special) vocab entry, so it decodes verbatim
        // rather than being stripped.
        assert_eq!(tok.decode(&[2]), "[UNK]");
    }

    #[test]
    fn decode_skips_special_tokens() {
        // The load-bearing flag: our decode passes skip_special_tokens=true, so a
        // special id (3 = "<|special|>" in the fixture) is dropped while the
        // content tokens survive — this is what keeps reward text clean.
        let tok = HfTokenizer::from_file(fixture()).unwrap();
        assert_eq!(tok.decode(&[3, 0, 1]), "hello world");
    }

    #[test]
    fn out_of_range_ids_decode_lossily_to_empty() {
        // Documented lossy-decode behavior: an id with no token is dropped (not an
        // error), so the wrapper returns an empty string via the Ok path.
        let tok = HfTokenizer::from_file(fixture()).unwrap();
        assert_eq!(tok.decode(&[999]), "");
    }

    #[test]
    fn from_file_rejects_a_missing_path() {
        let err = HfTokenizer::from_file("/no/such/tokenizer.json").unwrap_err();
        assert!(
            err.to_string().contains("failed to load tokenizer"),
            "expected a load error, got: {err}"
        );
    }
}
