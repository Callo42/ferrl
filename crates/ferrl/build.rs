//! Embed the exact clean Git revision used to build the operational CLI.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

const SOURCE_PATHS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config",
    ".cargo/config.toml",
    "crates/ferrl/Cargo.toml",
    "crates/ferrl/build.rs",
    "crates/ferrl/src",
];

const REQUIRED_SOURCE_FILES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "crates/ferrl/Cargo.toml",
    "crates/ferrl/build.rs",
];

struct HeadBlob {
    object_id: String,
    path: PathBuf,
}

fn git_command(directory: &Path) -> Command {
    const REPOSITORY_OVERRIDES: &[&str] = &[
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    ];
    let mut command = Command::new("git");
    for variable in REPOSITORY_OVERRIDES {
        command.env_remove(variable);
    }
    command
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .arg("-C")
        .arg(directory);
    command
}

fn git_output(directory: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let output = git_command(directory).args(args).output().ok()?;
    output.status.success().then_some(output.stdout)
}

fn git_success(directory: &Path, args: &[&str]) -> bool {
    git_command(directory)
        .args(args)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn valid_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn git_path(directory: &Path, args: &[&str]) -> Option<PathBuf> {
    let value = String::from_utf8(git_output(directory, args)?).ok()?;
    let path = PathBuf::from(value.trim());
    Some(if path.is_absolute() {
        path
    } else {
        directory.join(path)
    })
}

fn watch(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}

#[allow(clippy::cognitive_complexity)] // one linear strict parser for Git's NUL wire format
fn parse_head_blobs(bytes: &[u8]) -> Option<Vec<HeadBlob>> {
    let mut blobs = Vec::new();
    for entry in bytes
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
    {
        let tab = entry.iter().position(|byte| *byte == b'\t')?;
        let (header, path) = entry.split_at(tab);
        let path = path.get(1..)?;
        let header = std::str::from_utf8(header).ok()?;
        let mut fields = header.split_ascii_whitespace();
        let mode = fields.next()?;
        let kind = fields.next()?;
        let object_id = fields.next()?;
        if fields.next().is_some()
            || kind != "blob"
            || !matches!(mode, "100644" | "100755")
            || !valid_object_id(object_id)
        {
            return None;
        }
        let path = PathBuf::from(std::str::from_utf8(path).ok()?);
        if path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return None;
        }
        blobs.push(HeadBlob {
            object_id: object_id.to_owned(),
            path,
        });
    }
    (!blobs.is_empty()).then_some(blobs)
}

fn source_path_args<'a>(prefix: &'a [&'a str], commit: Option<&'a str>) -> Vec<&'a str> {
    let mut args =
        Vec::with_capacity(prefix.len() + usize::from(commit.is_some()) + 1 + SOURCE_PATHS.len());
    args.extend_from_slice(prefix);
    if let Some(commit) = commit {
        args.push(commit);
    }
    args.push("--");
    args.extend_from_slice(SOURCE_PATHS);
    args
}

fn commit_source_blobs(repo_root: &Path, commit: &str) -> Option<Vec<HeadBlob>> {
    let args = source_path_args(&["ls-tree", "-r", "-z"], Some(commit));
    let bytes = git_output(repo_root, &args)?;
    let blobs = parse_head_blobs(&bytes)?;
    REQUIRED_SOURCE_FILES
        .iter()
        .all(|required| blobs.iter().any(|blob| blob.path == Path::new(required)))
        .then_some(())?;
    blobs
        .iter()
        .any(|blob| blob.path.starts_with("crates/ferrl/src"))
        .then_some(blobs)
}

fn worktree_blobs_match(repo_root: &Path, blobs: &[HeadBlob]) -> bool {
    if blobs.iter().any(
        |blob| match std::fs::symlink_metadata(repo_root.join(&blob.path)) {
            Ok(metadata) => !metadata.file_type().is_file(),
            Err(_) => true,
        },
    ) {
        return false;
    }

    blobs.chunks(128).all(|chunk| {
        let output = git_command(repo_root)
            .args(["hash-object", "--no-filters", "--"])
            .args(chunk.iter().map(|blob| &blob.path))
            .output();
        let Ok(output) = output else {
            return false;
        };
        if !output.status.success() {
            return false;
        }
        let hashes = output
            .stdout
            .split(|byte| *byte == b'\n')
            .filter(|hash| !hash.is_empty())
            .collect::<Vec<_>>();
        hashes.len() == chunk.len()
            && hashes
                .iter()
                .zip(chunk)
                .all(|(actual, expected)| *actual == expected.object_id.as_bytes())
    })
}

fn index_entries_are_ordinary(repo_root: &Path, blobs: &[HeadBlob]) -> bool {
    let args = source_path_args(&["ls-files", "-v", "-z"], None);
    let Some(bytes) = git_output(repo_root, &args) else {
        return false;
    };
    let entries = bytes
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();
    if entries.len() != blobs.len() || entries.iter().any(|entry| !entry.starts_with(b"H ")) {
        return false;
    }
    let Some(paths) = entries
        .iter()
        .map(|entry| std::str::from_utf8(&entry[2..]).ok().map(PathBuf::from))
        .collect::<Option<BTreeSet<_>>>()
    else {
        return false;
    };
    paths == blobs.iter().map(|blob| blob.path.clone()).collect()
}

fn resolve_head_commit(repo_root: &Path) -> Option<String> {
    git_output(repo_root, &["rev-parse", "--verify", "HEAD^{commit}"])
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| valid_object_id(value))
}

/// Return whether every build-relevant source byte and index entry equals `commit`.
pub(crate) fn source_tree_is_exact_at(repo_root: &Path, commit: &str) -> bool {
    if !valid_object_id(commit) {
        return false;
    }
    let Some(blobs) = commit_source_blobs(repo_root, commit) else {
        return false;
    };
    let diff_args = source_path_args(&["diff-index", "--cached", "--quiet"], Some(commit));
    let untracked_args = source_path_args(&["ls-files", "--others", "-z"], None);
    worktree_blobs_match(repo_root, &blobs)
        && index_entries_are_ordinary(repo_root, &blobs)
        && git_success(repo_root, &diff_args)
        && git_output(repo_root, &untracked_args).is_some_and(|untracked| untracked.is_empty())
}

/// Return whether every build-relevant source byte and index entry equals the one
/// commit currently named by `HEAD`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn source_tree_is_exact(repo_root: &Path) -> bool {
    resolve_head_commit(repo_root).is_some_and(|commit| source_tree_is_exact_at(repo_root, &commit))
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SourceIdentity {
    pub(crate) commit: Option<String>,
    pub(crate) dirty: bool,
}

fn canonical_path(path: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(path).ok()
}

/// Resolve and validate one immutable commit identity for the expected workspace.
/// The hook exists only so the regression suite can deterministically move `HEAD`
/// across the final recheck; production supplies a no-op closure.
pub(crate) fn inspect_source_identity_with_hook(
    manifest_dir: &Path,
    after_validation: impl FnOnce(),
) -> SourceIdentity {
    let expected_root = canonical_path(&manifest_dir.join("../.."));
    let discovered_root = git_path(manifest_dir, &["rev-parse", "--show-toplevel"])
        .as_deref()
        .and_then(canonical_path);
    let Some(repo_root) =
        expected_root.filter(|expected| Some(expected) == discovered_root.as_ref())
    else {
        return SourceIdentity {
            commit: None,
            dirty: true,
        };
    };
    let Some(commit) = resolve_head_commit(&repo_root) else {
        return SourceIdentity {
            commit: None,
            dirty: true,
        };
    };
    let exact = source_tree_is_exact_at(&repo_root, &commit);
    after_validation();
    if resolve_head_commit(&repo_root).as_deref() != Some(commit.as_str()) {
        return SourceIdentity {
            commit: None,
            dirty: true,
        };
    }
    SourceIdentity {
        commit: Some(commit),
        dirty: !exact,
    }
}

fn main() {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    let identity = inspect_source_identity_with_hook(&manifest_dir, || {});
    let commit = identity.commit.unwrap_or_else(|| "unknown".to_owned());
    let dirty = identity.dirty;

    println!("cargo:rustc-env=FERRL_BUILD_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=FERRL_BUILD_GIT_DIRTY={dirty}");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../../Cargo.toml");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../rust-toolchain.toml");
    println!("cargo:rerun-if-changed=../../.cargo/config");
    println!("cargo:rerun-if-changed=../../.cargo/config.toml");
    if let Some(git_dir) = git_path(&manifest_dir, &["rev-parse", "--absolute-git-dir"]) {
        watch(&git_dir.join("HEAD"));
        watch(&git_dir.join("index"));
    }
    if let Some(common_dir) = git_path(&manifest_dir, &["rev-parse", "--git-common-dir"]) {
        if let Some(symbolic_ref) = git_output(&manifest_dir, &["symbolic-ref", "-q", "HEAD"])
            .and_then(|bytes| String::from_utf8(bytes).ok())
        {
            watch(&common_dir.join(symbolic_ref.trim()));
        }
        watch(&common_dir.join("refs/heads"));
        watch(&common_dir.join("packed-refs"));
    }
}
