//! Regression tests for the build-time source-to-commit binding.

use std::path::{Path, PathBuf};
use std::process::Command;

// The included build script must emit Cargo directives on stdout; this test calls only
// its exact-source helper.
#[allow(dead_code, clippy::print_stdout)]
#[path = "../build.rs"]
mod build_script;

struct TempRepo(PathBuf);

impl TempRepo {
    fn new() -> Self {
        let unique = format!(
            "ferrl-build-source-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(root.join("crates/ferrl/src")).unwrap();
        Self(root)
    }

    fn seeded() -> Self {
        let repo = Self::new();
        repo.write("Cargo.toml", b"[workspace]\nmembers = [\"crates/ferrl\"]\n");
        repo.write("Cargo.lock", b"# exact lock\n");
        repo.write(
            "rust-toolchain.toml",
            b"[toolchain]\nchannel = \"stable\"\n",
        );
        std::fs::create_dir_all(repo.path().join(".cargo")).unwrap();
        repo.write(".cargo/config.toml", b"[build]\ntarget-dir = \"target\"\n");
        repo.write(
            "crates/ferrl/Cargo.toml",
            b"[package]\nname = \"ferrl\"\nversion = \"0.0.0\"\n",
        );
        repo.write("crates/ferrl/build.rs", b"fn main() {}\n");
        repo.write("crates/ferrl/src/lib.rs", b"pub fn exact() {}\n");
        assert!(repo.git(&["init", "--quiet"]).status.success());
        assert!(repo.git(&["add", "."]).status.success());
        assert!(repo
            .git(&[
                "-c",
                "user.name=Ferrl Test",
                "-c",
                "user.email=ferrl@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "--no-verify",
                "-m",
                "test: seed source",
            ])
            .status
            .success());
        repo
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn git(&self, args: &[&str]) -> std::process::Output {
        Command::new("git")
            .arg("-C")
            .arg(&self.0)
            .args(args)
            .output()
            .unwrap()
    }

    fn write(&self, relative: &str, bytes: &[u8]) {
        if let Some(parent) = self.0.join(relative).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(self.0.join(relative), bytes).unwrap();
    }

    fn head(&self) -> String {
        String::from_utf8(self.git(&["rev-parse", "HEAD^{commit}"]).stdout)
            .unwrap()
            .trim()
            .to_owned()
    }

    fn manifest_dir(&self) -> PathBuf {
        self.path().join("crates/ferrl")
    }
}

impl Drop for TempRepo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // one ordered mutation-sensitive Git control
fn exact_source_check_detects_assume_unchanged_modification() {
    let repo = TempRepo::seeded();
    assert!(build_script::source_tree_is_exact(repo.path()));

    assert!(repo
        .git(&[
            "update-index",
            "--assume-unchanged",
            "crates/ferrl/src/lib.rs",
        ])
        .status
        .success());
    assert!(
        !build_script::source_tree_is_exact(repo.path()),
        "assume-unchanged must be rejected even before content diverges"
    );
    repo.write("crates/ferrl/src/lib.rs", b"pub fn forged() {}\n");
    assert!(repo.git(&["status", "--porcelain"]).stdout.is_empty());
    assert!(!build_script::source_tree_is_exact(repo.path()));
}

#[test]
#[allow(clippy::cognitive_complexity)] // one ordered replacement-ref attack control
fn exact_source_check_ignores_git_replacement_objects() {
    let repo = TempRepo::seeded();
    let original = String::from_utf8(repo.git(&["rev-parse", "HEAD"]).stdout)
        .unwrap()
        .trim()
        .to_owned();
    repo.write("crates/ferrl/src/lib.rs", b"pub fn forged() {}\n");
    assert!(repo.git(&["add", "."]).status.success());
    assert!(repo
        .git(&[
            "-c",
            "user.name=Ferrl Test",
            "-c",
            "user.email=ferrl@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--no-verify",
            "-m",
            "test: replacement source",
        ])
        .status
        .success());
    let replacement = String::from_utf8(repo.git(&["rev-parse", "HEAD"]).stdout)
        .unwrap()
        .trim()
        .to_owned();
    assert!(repo
        .git(&["replace", &original, &replacement])
        .status
        .success());
    assert!(repo
        .git(&["update-ref", "HEAD", &original])
        .status
        .success());
    assert_eq!(
        String::from_utf8(repo.git(&["rev-parse", "HEAD"]).stdout)
            .unwrap()
            .trim(),
        original
    );
    assert!(
        repo.git(&["status", "--porcelain"]).stdout.is_empty(),
        "replacement refs should fool Git's ordinary cleanliness view"
    );
    assert!(!build_script::source_tree_is_exact(repo.path()));
}

#[test]
fn exact_source_check_binds_toolchain_and_repository_cargo_config() {
    let repo = TempRepo::seeded();
    let head = repo.head();
    assert!(build_script::source_tree_is_exact_at(repo.path(), &head));

    repo.write(
        "rust-toolchain.toml",
        b"[toolchain]\nchannel = \"nightly\"\n",
    );
    assert!(!build_script::source_tree_is_exact_at(repo.path(), &head));
    assert!(repo
        .git(&["checkout-index", "--force", "--", "rust-toolchain.toml"])
        .status
        .success());
    assert!(build_script::source_tree_is_exact_at(repo.path(), &head));

    repo.write(
        ".cargo/config.toml",
        b"[build]\nrustflags = [\"-Copt-level=3\"]\n",
    );
    assert!(!build_script::source_tree_is_exact_at(repo.path(), &head));
}

#[test]
#[allow(clippy::cognitive_complexity)] // one ordered optional-input/index-flag control
fn exact_source_check_rejects_untracked_cargo_config_and_skip_worktree() {
    let repo = TempRepo::seeded();
    let head = repo.head();
    assert!(build_script::source_tree_is_exact_at(repo.path(), &head));

    repo.write(
        ".cargo/config",
        b"[build]\nrustflags = [\"-Cdebuginfo=0\"]\n",
    );
    assert!(
        !build_script::source_tree_is_exact_at(repo.path(), &head),
        "an untracked repository-local Cargo config must invalidate provenance"
    );
    std::fs::remove_file(repo.path().join(".cargo/config")).unwrap();
    assert!(build_script::source_tree_is_exact_at(repo.path(), &head));

    assert!(repo
        .git(&["update-index", "--skip-worktree", ".cargo/config.toml",])
        .status
        .success());
    assert!(
        !build_script::source_tree_is_exact_at(repo.path(), &head),
        "skip-worktree must be rejected even before content diverges"
    );
    repo.write(
        ".cargo/config.toml",
        b"[build]\nrustflags = [\"-Copt-level=3\"]\n",
    );
    assert!(repo.git(&["status", "--porcelain"]).stdout.is_empty());
    assert!(!build_script::source_tree_is_exact_at(repo.path(), &head));
}

#[test]
fn exact_source_check_rejects_member_and_ancestor_build_config_shadows() {
    const SHADOWS: [(&str, &[u8]); 9] = [
        ("rust-toolchain", b"nightly\n" as &[u8]),
        ("crates/rust-toolchain", b"nightly\n" as &[u8]),
        (
            "crates/rust-toolchain.toml",
            b"[toolchain]\nchannel = \"nightly\"\n" as &[u8],
        ),
        (
            "crates/.cargo/config",
            b"[build]\nrustflags = [\"-Copt-level=3\"]\n" as &[u8],
        ),
        (
            "crates/.cargo/config.toml",
            b"[build]\nrustflags = [\"-Copt-level=3\"]\n" as &[u8],
        ),
        ("crates/ferrl/rust-toolchain", b"nightly\n" as &[u8]),
        (
            "crates/ferrl/rust-toolchain.toml",
            b"[toolchain]\nchannel = \"nightly\"\n" as &[u8],
        ),
        (
            "crates/ferrl/.cargo/config",
            b"[build]\nrustflags = [\"-Copt-level=3\"]\n" as &[u8],
        ),
        (
            "crates/ferrl/.cargo/config.toml",
            b"[build]\nrustflags = [\"-Copt-level=3\"]\n" as &[u8],
        ),
    ];

    for (path, bytes) in SHADOWS {
        let repo = TempRepo::seeded();
        let head = repo.head();
        assert!(build_script::source_tree_is_exact_at(repo.path(), &head));
        repo.write(path, bytes);
        assert!(
            !build_script::source_tree_is_exact_at(repo.path(), &head),
            "untracked build-configuration shadow {path} must invalidate provenance"
        );
    }
}

#[test]
fn build_identity_rejects_non_commit_head() {
    let repo = TempRepo::seeded();
    let tree = String::from_utf8(repo.git(&["rev-parse", "HEAD^{tree}"]).stdout)
        .unwrap()
        .trim()
        .to_owned();
    std::fs::write(repo.path().join(".git/HEAD"), format!("{tree}\n")).unwrap();
    assert!(!build_script::source_tree_is_exact(repo.path()));
    let identity = build_script::inspect_source_identity_with_hook(&repo.manifest_dir(), || {});
    assert_eq!(identity.commit, None);
    assert!(identity.dirty);
}

#[test]
#[allow(clippy::cognitive_complexity)] // one deterministic validate-then-ref-drift control
fn build_identity_rejects_head_drift_after_validation() {
    let repo = TempRepo::seeded();
    let original = repo.head();
    repo.write("crates/ferrl/src/lib.rs", b"pub fn second() {}\n");
    assert!(repo.git(&["add", "."]).status.success());
    assert!(repo
        .git(&[
            "-c",
            "user.name=Ferrl Test",
            "-c",
            "user.email=ferrl@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--no-verify",
            "-m",
            "test: drift target",
        ])
        .status
        .success());
    let replacement = repo.head();
    assert!(repo
        .git(&["update-ref", "HEAD", &original])
        .status
        .success());
    assert!(repo.git(&["read-tree", &original]).status.success());
    assert!(repo
        .git(&["checkout-index", "--all", "--force"])
        .status
        .success());

    let identity = build_script::inspect_source_identity_with_hook(&repo.manifest_dir(), || {
        assert!(repo
            .git(&["update-ref", "HEAD", &replacement])
            .status
            .success());
    });
    assert_eq!(identity.commit, None);
    assert!(identity.dirty);
}

#[test]
fn build_identity_rejects_nested_repository_root() {
    let repo = TempRepo::seeded();
    let nested = repo.manifest_dir();
    assert!(Command::new("git")
        .arg("-C")
        .arg(&nested)
        .args(["init", "--quiet"])
        .status()
        .unwrap()
        .success());

    std::fs::write(
        nested.join("src/lib.rs"),
        b"pub fn forged_outer_source() {}\n",
    )
    .unwrap();
    std::fs::write(
        nested.join("Cargo.toml"),
        b"[workspace]\nmembers = [\"crates/ferrl\"]\n",
    )
    .unwrap();
    std::fs::write(nested.join("Cargo.lock"), b"# clean decoy lock\n").unwrap();
    std::fs::write(
        nested.join("rust-toolchain.toml"),
        b"[toolchain]\nchannel = \"stable\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(nested.join(".cargo")).unwrap();
    std::fs::write(
        nested.join(".cargo/config.toml"),
        b"[build]\ntarget-dir = \"target\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(nested.join("crates/ferrl/src")).unwrap();
    std::fs::write(
        nested.join("crates/ferrl/Cargo.toml"),
        b"[package]\nname = \"ferrl\"\nversion = \"0.0.0\"\n",
    )
    .unwrap();
    std::fs::write(nested.join("crates/ferrl/build.rs"), b"fn main() {}\n").unwrap();
    std::fs::write(
        nested.join("crates/ferrl/src/lib.rs"),
        b"pub fn decoy_source() {}\n",
    )
    .unwrap();
    assert!(Command::new("git")
        .arg("-C")
        .arg(&nested)
        .args(["add", "."])
        .status()
        .unwrap()
        .success());
    assert!(Command::new("git")
        .arg("-C")
        .arg(&nested)
        .args([
            "-c",
            "user.name=Ferrl Test",
            "-c",
            "user.email=ferrl@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--no-verify",
            "-m",
            "test: seed complete decoy",
        ])
        .status()
        .unwrap()
        .success());
    assert!(build_script::source_tree_is_exact(&nested));

    let identity = build_script::inspect_source_identity_with_hook(&nested, || {});
    assert_eq!(identity.commit, None);
    assert!(identity.dirty);
}
