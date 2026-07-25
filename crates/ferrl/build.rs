//! Embed the exact clean Git revision used to build the operational CLI.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

const SOURCE_PATHS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "crates/ferrl/Cargo.toml",
    "crates/ferrl/build.rs",
    "crates/ferrl/src",
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

fn head_source_blobs(repo_root: &Path) -> Option<Vec<HeadBlob>> {
    let bytes = git_output(
        repo_root,
        &[
            "ls-tree",
            "-r",
            "-z",
            "HEAD",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "crates/ferrl/Cargo.toml",
            "crates/ferrl/build.rs",
            "crates/ferrl/src",
        ],
    )?;
    let blobs = parse_head_blobs(&bytes)?;
    SOURCE_PATHS
        .iter()
        .take(4)
        .all(|required| blobs.iter().any(|blob| blob.path == Path::new(required)))
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
    let Some(bytes) = git_output(
        repo_root,
        &[
            "ls-files",
            "-v",
            "-z",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "crates/ferrl/Cargo.toml",
            "crates/ferrl/build.rs",
            "crates/ferrl/src",
        ],
    ) else {
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

/// Return whether every build-relevant source byte and index entry equals `HEAD`.
pub(crate) fn source_tree_is_exact(repo_root: &Path) -> bool {
    let Some(blobs) = head_source_blobs(repo_root) else {
        return false;
    };
    worktree_blobs_match(repo_root, &blobs)
        && index_entries_are_ordinary(repo_root, &blobs)
        && git_success(
            repo_root,
            &[
                "diff-index",
                "--cached",
                "--quiet",
                "HEAD",
                "--",
                "Cargo.toml",
                "Cargo.lock",
                "crates/ferrl/Cargo.toml",
                "crates/ferrl/build.rs",
                "crates/ferrl/src",
            ],
        )
        && git_output(
            repo_root,
            &[
                "ls-files",
                "--others",
                "-z",
                "--",
                "Cargo.toml",
                "Cargo.lock",
                "crates/ferrl/Cargo.toml",
                "crates/ferrl/build.rs",
                "crates/ferrl/src",
            ],
        )
        .is_some_and(|untracked| untracked.is_empty())
}

fn main() {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    let repo_root = git_path(&manifest_dir, &["rev-parse", "--show-toplevel"]);
    let commit = repo_root
        .as_deref()
        .and_then(|root| git_output(root, &["rev-parse", "--verify", "HEAD"]))
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| valid_object_id(value))
        .unwrap_or_else(|| "unknown".to_owned());
    let dirty = repo_root
        .as_deref()
        .is_none_or(|root| !source_tree_is_exact(root));

    println!("cargo:rustc-env=FERRL_BUILD_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=FERRL_BUILD_GIT_DIRTY={dirty}");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../../Cargo.toml");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
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
