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
        std::fs::write(self.0.join(relative), bytes).unwrap();
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
