//! Staged execution backends for sealed verifier assets.
//!
//! [`SameUidApptainerSandbox`] is the no-administrator training backend. It keeps
//! execution under the caller's non-root UID, but still copies kernel-sealed
//! descriptors into a fresh private request directory before launching
//! Apptainer. [`VerifierExecutorSandbox`] sends the same path-free request and
//! descriptors over an authenticated Unix socket to a dedicated non-root UID.
//! Both paths share request validation, read-only asset staging, fresh writable
//! scratch, supervision, and no-follow cleanup; they differ explicitly in their
//! host-UID assurance boundary and never fall back to one another. The same-UID
//! backend constrains the candidate launched through it, but cannot defend against
//! an arbitrary malicious peer process already running under the caller's host UID.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::sandbox::{
    Bind, BindMode, NetworkPolicy, ProtectedOutput, ResourceLimits, RunOutcome, RunSpec, RunStatus,
    Sandbox, SandboxError,
};

/// Default protected verifier executor socket.
pub const DEFAULT_VERIFIER_EXECUTOR_SOCKET: &str = "/run/ferrl/verifier-executor.sock";
/// Schema version for serialized verifier isolation preflight evidence.
pub const VERIFIER_ISOLATION_EVIDENCE_VERSION: u32 = 1;
const SAME_UID_WORK_ROOT_PREFIX: &str = "ferrl-verifier";

/// Versioned host-identity boundary used for one verifier launch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierIsolationTier {
    /// Hardened Apptainer execution under the training process's non-root host UID.
    /// This does not resist an arbitrary hostile peer under that same UID.
    #[default]
    SameUidApptainerV1,
    /// Execution delegated to a protected service under a distinct non-root host UID.
    DedicatedUidServiceV1,
}

impl VerifierIsolationTier {
    /// Stable provenance identifier for this tier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SameUidApptainerV1 => "same_uid_apptainer_v1",
            Self::DedicatedUidServiceV1 => "dedicated_uid_service_v1",
        }
    }
}

/// Relationship between the verifier launcher and training process host UIDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierUidBoundary {
    /// Both execute under the same non-root host UID.
    SameHostUid,
    /// The verifier service executes under a distinct non-root host UID.
    DistinctHostUid,
}

/// How sealed verifier descriptors reach private staged paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierAssetTransport {
    /// The caller copies its own sealed descriptors without a socket hop.
    InProcessSealedCopy,
    /// Sealed descriptors cross an authenticated Unix socket with `SCM_RIGHTS`.
    ScmRightsSealedCopy,
}

/// Canonical, serializable evidence produced by verifier backend preflight.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierIsolationEvidence {
    /// Evidence schema version.
    pub contract_version: u32,
    /// Versioned assurance tier selected by the caller.
    pub tier: VerifierIsolationTier,
    /// Host UID that requested verifier execution.
    pub requester_uid: u32,
    /// Host UID that launches Apptainer.
    pub launcher_uid: u32,
    /// Host-UID relationship provided by the backend.
    pub uid_boundary: VerifierUidBoundary,
    /// Descriptor transport used before the Apptainer launch.
    pub asset_transport: VerifierAssetTransport,
    /// Canonical absolute Apptainer executable path.
    pub apptainer_path: PathBuf,
    /// SHA-256 of the exact executable bytes observed during preflight.
    pub apptainer_sha256: String,
    /// Exact executable length observed during preflight.
    pub apptainer_len_bytes: u64,
    /// Bounded, whitespace-normalized output from `apptainer --version`.
    pub apptainer_version: String,
    /// Canonical private root used for staged assets and fresh scratch.
    pub work_root: PathBuf,
    /// UID owning `work_root` at preflight.
    pub work_root_uid: u32,
    /// Filesystem device containing `work_root` at preflight.
    pub work_root_device: u64,
    /// Filesystem inode of `work_root` at preflight.
    pub work_root_inode: u64,
    /// Exact permission bits on `work_root`.
    pub work_root_mode: u32,
}

/// In-process staged Apptainer execution under the caller's non-root UID.
///
/// `work_root` must be a normalized path. It is created atomically with mode `0700`
/// when absent; an existing root must be a non-symlink directory owned by the
/// effective UID with that exact mode. `apptainer_bin` must be an absolute,
/// root-owned, non-writable executable beneath root-owned, non-writable directories.
/// Every run copies sealed verifier assets into a new private request directory and
/// creates fresh scratch there; configured read-write source paths are never reused.
/// This tier does not claim isolation from arbitrary peer processes already running
/// under the same host UID.
#[derive(Debug, Clone)]
pub struct SameUidApptainerSandbox {
    work_root: PathBuf,
    apptainer_bin: PathBuf,
}

impl Default for SameUidApptainerSandbox {
    fn default() -> Self {
        Self::new(std::env::temp_dir().join(format!(
            "{SAME_UID_WORK_ROOT_PREFIX}-{}",
            std::process::id()
        )))
    }
}

impl SameUidApptainerSandbox {
    /// Construct a same-UID backend using `/usr/bin/apptainer`.
    #[must_use]
    pub fn new(work_root: impl Into<PathBuf>) -> Self {
        Self {
            work_root: work_root.into(),
            apptainer_bin: PathBuf::from("/usr/bin/apptainer"),
        }
    }

    /// Override the absolute trusted Apptainer executable.
    #[must_use]
    pub fn with_apptainer_bin(mut self, bin: impl Into<PathBuf>) -> Self {
        self.apptainer_bin = bin.into();
        self
    }

    /// Return the private root used for per-request staging and scratch.
    #[must_use]
    pub fn work_root(&self) -> &Path {
        &self.work_root
    }

    /// Return the configured Apptainer executable.
    #[must_use]
    pub fn apptainer_bin(&self) -> &Path {
        &self.apptainer_bin
    }

    /// Return the backend's explicit assurance tier.
    #[must_use]
    pub const fn isolation_tier(&self) -> VerifierIsolationTier {
        VerifierIsolationTier::SameUidApptainerV1
    }

    /// Create or validate the private work root and validate the local UID and
    /// trusted Apptainer executable.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError::InvalidSpec`] when this backend cannot provide its
    /// declared same-UID boundary. No executor socket or alternate backend is tried.
    pub fn preflight(&self) -> Result<VerifierIsolationEvidence, SandboxError> {
        #[cfg(target_os = "linux")]
        {
            linux::preflight_same_uid(self)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(SandboxError::InvalidSpec(
                "same-UID verifier execution requires Linux".to_string(),
            ))
        }
    }
}

impl Sandbox for SameUidApptainerSandbox {
    fn run(&self, spec: &RunSpec) -> Result<RunOutcome, SandboxError> {
        #[cfg(target_os = "linux")]
        {
            linux::run_same_uid(self, spec)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = spec;
            Err(SandboxError::InvalidSpec(
                "same-UID verifier execution requires Linux".to_string(),
            ))
        }
    }
}

/// Client for a protected verifier executor running under a dedicated UID.
#[derive(Debug, Clone)]
pub struct VerifierExecutorSandbox {
    socket_path: PathBuf,
}

impl Default for VerifierExecutorSandbox {
    fn default() -> Self {
        Self::new(DEFAULT_VERIFIER_EXECUTOR_SOCKET)
    }
}

impl VerifierExecutorSandbox {
    /// Connect to the executor at `socket_path`.
    #[must_use]
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }

    /// Return the configured executor socket path.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Return the backend's explicit assurance tier.
    #[must_use]
    pub const fn isolation_tier(&self) -> VerifierIsolationTier {
        VerifierIsolationTier::DedicatedUidServiceV1
    }

    /// Request service-produced isolation evidence over the authenticated socket.
    ///
    /// # Errors
    ///
    /// Returns a fail-closed [`SandboxError`] if the endpoint, peer UID, protocol,
    /// or service-produced dedicated-tier evidence is invalid. No local backend is
    /// attempted.
    pub fn preflight(&self) -> Result<VerifierIsolationEvidence, SandboxError> {
        #[cfg(target_os = "linux")]
        {
            linux::preflight_client(&self.socket_path)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(SandboxError::InvalidSpec(
                "protected verifier execution requires Linux".to_string(),
            ))
        }
    }
}

impl Sandbox for VerifierExecutorSandbox {
    fn run(&self, spec: &RunSpec) -> Result<RunOutcome, SandboxError> {
        #[cfg(target_os = "linux")]
        {
            linux::run_client(&self.socket_path, spec)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = spec;
            Err(SandboxError::InvalidSpec(
                "protected verifier execution requires Linux".to_string(),
            ))
        }
    }
}

/// Configuration for the long-running protected verifier executor.
#[derive(Debug, Clone)]
pub struct VerifierExecutorConfig {
    /// Unix socket clients connect to.
    pub socket_path: PathBuf,
    /// Pre-existing service-private root for per-request assets and scratch
    /// directories. It must be owned by the executor UID with mode `0700`.
    pub work_root: PathBuf,
    /// The only training-process UID accepted by `SO_PEERCRED`.
    pub client_uid: u32,
    /// Absolute, root-owned Apptainer executable used by the service.
    pub apptainer_bin: PathBuf,
    /// Socket permission bits, normally `0o660` with an administrator-managed
    /// service group shared with the training UID.
    pub socket_mode: u32,
}

impl VerifierExecutorConfig {
    /// Construct a service configuration with `/usr/bin/apptainer` and a
    /// group-accessible, non-world-accessible socket.
    #[must_use]
    pub fn new(
        socket_path: impl Into<PathBuf>,
        work_root: impl Into<PathBuf>,
        client_uid: u32,
    ) -> Self {
        Self {
            socket_path: socket_path.into(),
            work_root: work_root.into(),
            client_uid,
            apptainer_bin: PathBuf::from("/usr/bin/apptainer"),
            socket_mode: 0o660,
        }
    }

    /// Override the Apptainer executable.
    #[must_use]
    pub fn with_apptainer_bin(mut self, bin: impl Into<PathBuf>) -> Self {
        self.apptainer_bin = bin.into();
        self
    }

    /// Override the socket permission bits. World permissions are rejected when
    /// the service starts.
    #[must_use]
    pub fn with_socket_mode(mut self, mode: u32) -> Self {
        self.socket_mode = mode;
        self
    }
}

/// Fatal failure while starting or serving the protected executor.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerifierExecutorError {
    /// The service is only implemented on Linux.
    #[error("protected verifier executor requires Linux")]
    Unsupported,
    /// The deployment configuration violates the dedicated-UID boundary.
    #[error("invalid protected verifier executor configuration: {0}")]
    InvalidConfig(String),
    /// An operating-system operation failed.
    #[error("protected verifier executor I/O failed during {operation}: {source}")]
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
}

/// Run the protected executor until its listening socket fails.
///
/// The process must already be launched under a dedicated non-root service UID.
/// The socket parent is administrator-managed; it must be owned by the service
/// UID and must not be world-writable. `work_root` must already be service-owned
/// with mode `0700`; the executor does not create deployment directories.
///
/// # Errors
///
/// Returns [`VerifierExecutorError`] when deployment preflight or the listening
/// socket fails. Individual malformed or failed requests receive a structured
/// error response without stopping the service.
pub fn serve_verifier_executor(
    config: &VerifierExecutorConfig,
) -> Result<(), VerifierExecutorError> {
    #[cfg(target_os = "linux")]
    {
        linux::serve(config)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = config;
        Err(VerifierExecutorError::Unsupported)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ExecutorRequest {
    version: u32,
    image_fd: usize,
    command: Vec<String>,
    binds: Vec<ExecutorBind>,
    workdir: PathBuf,
    env: Vec<(String, String)>,
    gpu: bool,
    network: NetworkPolicy,
    limits: ResourceLimits,
    protected_output: Option<ExecutorProtectedOutput>,
    asset_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
enum ExecutorWireRequest {
    Preflight { version: u32 },
    Run(Box<ExecutorRequest>),
}

#[derive(Debug, Serialize, Deserialize)]
struct ExecutorBind {
    source: ExecutorBindSource,
    dst: PathBuf,
    mode: BindMode,
    total_limit: Option<u64>,
    directories: Vec<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
enum ExecutorBindSource {
    SealedAsset { fd_index: usize },
    FreshScratch,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExecutorProtectedOutput {
    bind_index: usize,
    relative_path: PathBuf,
    sandbox_socket: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
enum ExecutorResponse {
    Preflight {
        service_uid: u32,
        evidence: VerifierIsolationEvidence,
    },
    Outcome {
        service_uid: u32,
        outcome: RunOutcome,
    },
    Error(ExecutorWireError),
}

#[derive(Debug, Serialize, Deserialize)]
enum ExecutorWireError {
    InvalidSpec(String),
    Infrastructure { status: RunStatus, stderr: String },
    Executor(String),
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs::{self, File, OpenOptions};
    use std::io::{IoSlice, IoSliceMut, Read as _, Seek as _, SeekFrom, Write as _};
    use std::mem::MaybeUninit;
    use std::os::fd::{AsFd as _, OwnedFd};
    use std::os::unix::fs::{
        DirBuilderExt as _, FileTypeExt as _, MetadataExt as _, OpenOptionsExt as _,
        PermissionsExt as _,
    };
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::Duration;

    use rustix::net::{
        recvmsg, sendmsg, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags,
        SendAncillaryBuffer, SendAncillaryMessage, SendFlags,
    };
    use rustix::process::DumpableBehavior;
    use sha2::{Digest as _, Sha256};

    use super::*;
    use crate::sandbox::{
        is_owned_descriptor_path, validate_protected_output_mapping, ApptainerSandbox, CAPTURE_CAP,
    };

    const EXECUTOR_PROTOCOL_VERSION: u32 = 2;
    const MAX_ASSETS: usize = 32;
    const MAX_BINDS: usize = 32;
    const MAX_REQUEST_BYTES: usize = 1024 * 1024;
    const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
    // Three captures at six JSON bytes per escaped input byte, plus frame metadata.
    const _: () = assert!(CAPTURE_CAP * 6 * 3 + 1024 * 1024 <= MAX_RESPONSE_BYTES);
    const SCRATCH_DIRECTORY_ENV: [&str; 2] = ["HOME", "TRITON_CACHE_DIR"];
    const EXECUTOR_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
    const EXECUTOR_WIRE_GRACE: Duration = Duration::from_secs(30);
    const REQUIRED_SEALS: rustix::fs::SealFlags = rustix::fs::SealFlags::WRITE
        .union(rustix::fs::SealFlags::GROW)
        .union(rustix::fs::SealFlags::SHRINK)
        .union(rustix::fs::SealFlags::SEAL);
    static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn executor_wire_timeout(wall: Duration) -> Duration {
        wall.max(Duration::from_secs(1))
            .saturating_add(EXECUTOR_WIRE_GRACE)
    }

    pub(super) fn run_same_uid(
        sandbox: &SameUidApptainerSandbox,
        spec: &RunSpec,
    ) -> Result<RunOutcome, SandboxError> {
        let _ = preflight_same_uid(sandbox)?;
        let (request, assets) = request_from_spec(spec)?;
        let fds = assets.into_iter().map(OwnedFd::from).collect();
        execute_request_inner(request, fds, &sandbox.work_root, &sandbox.apptainer_bin)
    }

    pub(super) fn preflight_same_uid(
        sandbox: &SameUidApptainerSandbox,
    ) -> Result<VerifierIsolationEvidence, SandboxError> {
        validate_same_uid_config(sandbox)
    }

    pub(super) fn run_client(
        socket_path: &Path,
        spec: &RunSpec,
    ) -> Result<RunOutcome, SandboxError> {
        let (request, assets) = request_from_spec(spec)?;
        let wire_timeout = executor_wire_timeout(spec.limits.wall);
        let (mut stream, service_uid) = connect_authenticated(socket_path, wire_timeout)?;
        let wire_request = ExecutorWireRequest::Run(Box::new(request));
        send_request(&mut stream, &wire_request, &assets)
            .map_err(|error| SandboxError::Executor(error.to_string()))?;
        match read_response(&mut stream)
            .map_err(|error| SandboxError::Executor(error.to_string()))?
        {
            ExecutorResponse::Outcome {
                service_uid: response_uid,
                outcome,
            } if response_uid == service_uid => Ok(outcome),
            ExecutorResponse::Outcome {
                service_uid: response_uid,
                ..
            }
            | ExecutorResponse::Preflight {
                service_uid: response_uid,
                ..
            } => Err(SandboxError::Executor(format!(
                "executor response UID {response_uid} or kind does not match authenticated run request for UID {service_uid}"
            ))),
            ExecutorResponse::Error(error) => Err(wire_error_to_sandbox(error)),
        }
    }

    pub(super) fn preflight_client(
        socket_path: &Path,
    ) -> Result<VerifierIsolationEvidence, SandboxError> {
        let (mut stream, service_uid) =
            connect_authenticated(socket_path, EXECUTOR_HANDSHAKE_TIMEOUT)?;
        send_request(
            &mut stream,
            &ExecutorWireRequest::Preflight {
                version: EXECUTOR_PROTOCOL_VERSION,
            },
            &[],
        )
        .map_err(|error| SandboxError::Executor(error.to_string()))?;
        match read_response(&mut stream)
            .map_err(|error| SandboxError::Executor(error.to_string()))?
        {
            ExecutorResponse::Preflight {
                service_uid: response_uid,
                evidence,
            } if response_uid == service_uid => {
                validate_dedicated_evidence(&evidence, service_uid)?;
                Ok(evidence)
            }
            ExecutorResponse::Preflight {
                service_uid: response_uid,
                ..
            }
            | ExecutorResponse::Outcome {
                service_uid: response_uid,
                ..
            } => Err(SandboxError::Executor(format!(
                "executor response UID {response_uid} or kind does not match authenticated preflight request for UID {service_uid}"
            ))),
            ExecutorResponse::Error(error) => Err(wire_error_to_sandbox(error)),
        }
    }

    fn connect_authenticated(
        socket_path: &Path,
        timeout: Duration,
    ) -> Result<(UnixStream, u32), SandboxError> {
        let service_uid = validate_client_socket(socket_path)?;
        let stream = UnixStream::connect(socket_path).map_err(|source| {
            SandboxError::Executor(format!(
                "could not connect to {}: {source}",
                socket_path.display()
            ))
        })?;
        stream.set_read_timeout(Some(timeout)).map_err(|source| {
            SandboxError::Executor(format!("could not set executor read deadline: {source}"))
        })?;
        stream.set_write_timeout(Some(timeout)).map_err(|source| {
            SandboxError::Executor(format!("could not set executor write deadline: {source}"))
        })?;
        let peer = rustix::net::sockopt::socket_peercred(&stream).map_err(|source| {
            SandboxError::Executor(format!("could not authenticate executor peer: {source}"))
        })?;
        if peer.uid.as_raw() != service_uid {
            return Err(SandboxError::Executor(format!(
                "executor peer UID {} does not own socket {} (UID {service_uid})",
                peer.uid.as_raw(),
                socket_path.display()
            )));
        }
        Ok((stream, service_uid))
    }

    fn validate_dedicated_evidence(
        evidence: &VerifierIsolationEvidence,
        service_uid: u32,
    ) -> Result<(), SandboxError> {
        let valid_hash = evidence.apptainer_sha256.len() == 64
            && evidence
                .apptainer_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if service_uid == 0
            || service_uid == rustix::process::geteuid().as_raw()
            || evidence.contract_version != VERIFIER_ISOLATION_EVIDENCE_VERSION
            || evidence.tier != VerifierIsolationTier::DedicatedUidServiceV1
            || evidence.requester_uid != rustix::process::geteuid().as_raw()
            || evidence.launcher_uid != service_uid
            || evidence.uid_boundary != VerifierUidBoundary::DistinctHostUid
            || evidence.asset_transport != VerifierAssetTransport::ScmRightsSealedCopy
            || !is_normal_absolute(&evidence.apptainer_path)
            || !is_normal_absolute(&evidence.work_root)
            || !valid_hash
            || evidence.apptainer_len_bytes == 0
            || evidence.apptainer_version.is_empty()
            || evidence.apptainer_version.len() > 512
            || evidence.work_root_uid != service_uid
            || evidence.work_root_inode == 0
            || evidence.work_root_mode != 0o700
        {
            return Err(SandboxError::Executor(
                "executor returned invalid dedicated-tier isolation evidence".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn serve(config: &VerifierExecutorConfig) -> Result<(), VerifierExecutorError> {
        let service_uid = validate_service_config(config)?;
        rustix::process::set_dumpable_behavior(DumpableBehavior::NotDumpable).map_err(
            |source| VerifierExecutorError::Io {
                operation: "disabling executor dumpability",
                source: source.into(),
            },
        )?;
        let listener = UnixListener::bind(&config.socket_path).map_err(|source| {
            VerifierExecutorError::Io {
                operation: "binding executor socket",
                source,
            }
        })?;
        fs::set_permissions(
            &config.socket_path,
            fs::Permissions::from_mode(config.socket_mode),
        )
        .map_err(|source| VerifierExecutorError::Io {
            operation: "setting executor socket permissions",
            source,
        })?;

        for accepted in listener.incoming() {
            let stream = accepted.map_err(|source| VerifierExecutorError::Io {
                operation: "accepting executor client",
                source,
            })?;
            let request_config = (*config).clone();
            let _request = thread::spawn(move || {
                let _ = handle_connection(stream, &request_config, service_uid);
            });
        }
        Err(VerifierExecutorError::InvalidConfig(
            "executor listener ended unexpectedly".to_string(),
        ))
    }

    fn validate_client_socket(socket_path: &Path) -> Result<u32, SandboxError> {
        if !socket_path.is_absolute() {
            return Err(SandboxError::InvalidSpec(
                "verifier executor socket must be absolute".to_string(),
            ));
        }
        let metadata = fs::symlink_metadata(socket_path).map_err(|source| {
            SandboxError::Executor(format!(
                "could not inspect executor socket {}: {source}",
                socket_path.display()
            ))
        })?;
        if !metadata.file_type().is_socket() {
            return Err(SandboxError::Executor(format!(
                "executor endpoint is not a Unix socket: {}",
                socket_path.display()
            )));
        }
        let client_uid = rustix::process::geteuid().as_raw();
        if client_uid == 0 || metadata.uid() == 0 || metadata.uid() == client_uid {
            return Err(SandboxError::Executor(
                "verifier executor client and socket owner must be distinct non-root UIDs"
                    .to_string(),
            ));
        }
        if metadata.mode() & 0o007 != 0 {
            return Err(SandboxError::Executor(
                "executor socket grants world permissions".to_string(),
            ));
        }
        let parent = socket_path.parent().ok_or_else(|| {
            SandboxError::InvalidSpec("executor socket has no parent directory".to_string())
        })?;
        let parent_metadata = fs::symlink_metadata(parent).map_err(|source| {
            SandboxError::Executor(format!(
                "could not inspect executor socket parent {}: {source}",
                parent.display()
            ))
        })?;
        if parent_metadata.file_type().is_symlink()
            || !parent_metadata.is_dir()
            || parent_metadata.uid() != metadata.uid()
            || parent_metadata.mode() & 0o027 != 0
        {
            return Err(SandboxError::Executor(
                "executor socket parent is not a protected service-owned directory".to_string(),
            ));
        }
        Ok(metadata.uid())
    }

    fn validate_service_config(
        config: &VerifierExecutorConfig,
    ) -> Result<u32, VerifierExecutorError> {
        let service_uid = rustix::process::geteuid().as_raw();
        if service_uid == 0 || config.client_uid == 0 || service_uid == config.client_uid {
            return Err(VerifierExecutorError::InvalidConfig(
                "executor and client must use distinct non-root UIDs".to_string(),
            ));
        }
        if !is_normal_absolute(&config.socket_path) || !is_normal_absolute(&config.work_root) {
            return Err(VerifierExecutorError::InvalidConfig(
                "socket_path and work_root must be absolute without '.' or '..'".to_string(),
            ));
        }
        validate_trusted_executor_binary(&config.apptainer_bin)?;
        if config.socket_mode & 0o007 != 0 || config.socket_mode & !0o777 != 0 {
            return Err(VerifierExecutorError::InvalidConfig(
                "socket_mode must contain permission bits with no world access".to_string(),
            ));
        }
        if config.socket_path.exists() {
            return Err(VerifierExecutorError::InvalidConfig(format!(
                "executor socket already exists: {}",
                config.socket_path.display()
            )));
        }
        validate_service_directory(
            config.socket_path.parent().ok_or_else(|| {
                VerifierExecutorError::InvalidConfig(
                    "executor socket has no parent directory".to_string(),
                )
            })?,
            service_uid,
            false,
            "socket parent",
        )?;

        validate_service_directory(&config.work_root, service_uid, true, "work root")?;
        service_isolation_evidence(config, service_uid)
            .map_err(|error| VerifierExecutorError::InvalidConfig(error.to_string()))?;
        Ok(service_uid)
    }

    fn validate_same_uid_config(
        sandbox: &SameUidApptainerSandbox,
    ) -> Result<VerifierIsolationEvidence, SandboxError> {
        let uid = rustix::process::geteuid().as_raw();
        if uid == 0 {
            return Err(SandboxError::InvalidSpec(
                "same-UID verifier must run under a non-root UID".to_string(),
            ));
        }
        let (canonical_work_root, metadata) = prepare_same_uid_work_root(&sandbox.work_root, uid)?;
        let binary =
            trusted_binary_identity(&sandbox.apptainer_bin, "same-UID verifier Apptainer")?;
        Ok(VerifierIsolationEvidence {
            contract_version: VERIFIER_ISOLATION_EVIDENCE_VERSION,
            tier: VerifierIsolationTier::SameUidApptainerV1,
            requester_uid: uid,
            launcher_uid: uid,
            uid_boundary: VerifierUidBoundary::SameHostUid,
            asset_transport: VerifierAssetTransport::InProcessSealedCopy,
            apptainer_path: binary.canonical_path,
            apptainer_sha256: binary.sha256,
            apptainer_len_bytes: binary.len_bytes,
            apptainer_version: binary.version,
            work_root: canonical_work_root,
            work_root_uid: metadata.uid(),
            work_root_device: metadata.dev(),
            work_root_inode: metadata.ino(),
            work_root_mode: metadata.mode() & 0o777,
        })
    }

    fn prepare_same_uid_work_root(
        work_root: &Path,
        uid: u32,
    ) -> Result<(PathBuf, fs::Metadata), SandboxError> {
        if !is_normal_absolute(work_root) {
            return Err(SandboxError::InvalidSpec(
                "same-UID verifier work_root must be absolute without '.' or '..'".to_string(),
            ));
        }
        match fs::symlink_metadata(work_root) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let mut builder = fs::DirBuilder::new();
                builder.mode(0o700);
                if let Err(error) = builder.create(work_root) {
                    if error.kind() != std::io::ErrorKind::AlreadyExists {
                        return Err(SandboxError::InvalidSpec(format!(
                            "could not create same-UID verifier work_root {}: {error}",
                            work_root.display()
                        )));
                    }
                }
            }
            Err(source) => {
                return Err(SandboxError::InvalidSpec(format!(
                    "could not inspect same-UID verifier work_root {}: {source}",
                    work_root.display()
                )))
            }
        }
        let canonical = fs::canonicalize(work_root).map_err(|source| {
            SandboxError::InvalidSpec(format!(
                "could not canonicalize same-UID verifier work_root {}: {source}",
                work_root.display()
            ))
        })?;
        if canonical != work_root {
            return Err(SandboxError::InvalidSpec(format!(
                "same-UID verifier work_root must already be canonical: configured {}, resolved {}",
                work_root.display(),
                canonical.display()
            )));
        }
        let metadata = fs::symlink_metadata(work_root).map_err(|source| {
            SandboxError::InvalidSpec(format!(
                "could not inspect same-UID verifier work_root {}: {source}",
                work_root.display()
            ))
        })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || metadata.uid() != uid
            || metadata.mode() & 0o777 != 0o700
        {
            return Err(SandboxError::InvalidSpec(format!(
                "same-UID verifier work_root {} must be a caller-owned non-symlink directory with mode 0700",
                work_root.display()
            )));
        }
        Ok((canonical, metadata))
    }

    struct TrustedBinaryIdentity {
        canonical_path: PathBuf,
        sha256: String,
        len_bytes: u64,
        version: String,
    }

    fn trusted_binary_identity(
        path: &Path,
        label: &str,
    ) -> Result<TrustedBinaryIdentity, SandboxError> {
        validate_trusted_executor_binary(path).map_err(|error| {
            SandboxError::InvalidSpec(format!("{label} executable is not trusted: {error}"))
        })?;
        validate_trusted_binary_parents(path, label)?;
        let canonical_path = fs::canonicalize(path).map_err(|source| {
            SandboxError::InvalidSpec(format!(
                "could not canonicalize {label} executable {}: {source}",
                path.display()
            ))
        })?;
        if canonical_path != path {
            return Err(SandboxError::InvalidSpec(format!(
                "{label} executable must already be canonical: configured {}, resolved {}",
                path.display(),
                canonical_path.display()
            )));
        }
        let (sha256, len_bytes) = hash_file(&canonical_path)?;
        let version = trusted_binary_version(&canonical_path)?;
        validate_trusted_executor_binary(&canonical_path).map_err(|error| {
            SandboxError::InvalidSpec(format!(
                "{label} executable changed during preflight: {error}"
            ))
        })?;
        let (after_sha256, after_len_bytes) = hash_file(&canonical_path)?;
        if sha256 != after_sha256 || len_bytes != after_len_bytes {
            return Err(SandboxError::InvalidSpec(format!(
                "{label} executable changed during preflight"
            )));
        }
        Ok(TrustedBinaryIdentity {
            canonical_path,
            sha256,
            len_bytes,
            version,
        })
    }

    fn hash_file(path: &Path) -> Result<(String, u64), SandboxError> {
        let mut file = File::open(path).map_err(|source| {
            SandboxError::InvalidSpec(format!(
                "could not open verifier executable {} for hashing: {source}",
                path.display()
            ))
        })?;
        let length = file
            .metadata()
            .map_err(|source| {
                SandboxError::InvalidSpec(format!(
                    "could not inspect verifier executable {} while hashing: {source}",
                    path.display()
                ))
            })?
            .len();
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer).map_err(|source| {
                SandboxError::InvalidSpec(format!(
                    "could not hash verifier executable {}: {source}",
                    path.display()
                ))
            })?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        Ok((format!("{:x}", hasher.finalize()), length))
    }

    fn trusted_binary_version(path: &Path) -> Result<String, SandboxError> {
        let output = std::process::Command::new(path)
            .arg("--version")
            .env_clear()
            .output()
            .map_err(|source| {
                SandboxError::InvalidSpec(format!(
                    "could not execute trusted verifier binary {} --version: {source}",
                    path.display()
                ))
            })?;
        if !output.status.success() {
            return Err(SandboxError::InvalidSpec(format!(
                "trusted verifier binary {} --version exited with {}",
                path.display(),
                output.status
            )));
        }
        let stdout = std::str::from_utf8(&output.stdout).map_err(|error| {
            SandboxError::InvalidSpec(format!(
                "trusted verifier binary {} emitted non-UTF-8 version output: {error}",
                path.display()
            ))
        })?;
        let stderr = std::str::from_utf8(&output.stderr).map_err(|error| {
            SandboxError::InvalidSpec(format!(
                "trusted verifier binary {} emitted non-UTF-8 version diagnostics: {error}",
                path.display()
            ))
        })?;
        let raw = if stdout.trim().is_empty() {
            stderr
        } else {
            stdout
        };
        let version = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if version.is_empty() || version.len() > 512 {
            return Err(SandboxError::InvalidSpec(format!(
                "trusted verifier binary {} emitted an empty or oversized version",
                path.display()
            )));
        }
        Ok(version)
    }

    fn validate_trusted_binary_parents(path: &Path, label: &str) -> Result<(), SandboxError> {
        let mut parent = path.parent();
        while let Some(directory) = parent {
            let metadata = fs::symlink_metadata(directory).map_err(|source| {
                SandboxError::InvalidSpec(format!(
                    "could not inspect {label} parent {}: {source}",
                    directory.display()
                ))
            })?;
            if metadata.file_type().is_symlink()
                || !metadata.is_dir()
                || metadata.uid() != 0
                || metadata.mode() & 0o022 != 0
            {
                return Err(SandboxError::InvalidSpec(format!(
                    "{label} parent {} must be a root-owned non-symlink directory with no group/world write access",
                    directory.display()
                )));
            }
            parent = directory.parent();
        }
        Ok(())
    }

    fn service_isolation_evidence(
        config: &VerifierExecutorConfig,
        service_uid: u32,
    ) -> Result<VerifierIsolationEvidence, SandboxError> {
        let observed_uid = rustix::process::geteuid().as_raw();
        if service_uid != observed_uid
            || service_uid == 0
            || config.client_uid == 0
            || service_uid == config.client_uid
        {
            return Err(SandboxError::Executor(
                "dedicated verifier evidence requires distinct observed non-root service and client UIDs"
                    .to_string(),
            ));
        }
        let (canonical_work_root, work_root_metadata) =
            service_work_root_identity(&config.work_root, service_uid)?;
        let binary = trusted_binary_identity(&config.apptainer_bin, "dedicated verifier Apptainer")
            .map_err(|error| SandboxError::Executor(error.to_string()))?;

        Ok(VerifierIsolationEvidence {
            contract_version: VERIFIER_ISOLATION_EVIDENCE_VERSION,
            tier: VerifierIsolationTier::DedicatedUidServiceV1,
            requester_uid: config.client_uid,
            launcher_uid: service_uid,
            uid_boundary: VerifierUidBoundary::DistinctHostUid,
            asset_transport: VerifierAssetTransport::ScmRightsSealedCopy,
            apptainer_path: binary.canonical_path,
            apptainer_sha256: binary.sha256,
            apptainer_len_bytes: binary.len_bytes,
            apptainer_version: binary.version,
            work_root: canonical_work_root,
            work_root_uid: work_root_metadata.uid(),
            work_root_device: work_root_metadata.dev(),
            work_root_inode: work_root_metadata.ino(),
            work_root_mode: work_root_metadata.mode() & 0o777,
        })
    }

    fn service_work_root_identity(
        work_root: &Path,
        service_uid: u32,
    ) -> Result<(PathBuf, fs::Metadata), SandboxError> {
        if !is_normal_absolute(work_root) {
            return Err(SandboxError::Executor(
                "dedicated verifier work_root must be an absolute normalized path".to_string(),
            ));
        }
        validate_service_directory(work_root, service_uid, true, "work root")
            .map_err(|error| SandboxError::Executor(error.to_string()))?;
        let canonical = fs::canonicalize(work_root).map_err(|source| {
            SandboxError::Executor(format!(
                "could not canonicalize dedicated verifier work_root {}: {source}",
                work_root.display()
            ))
        })?;
        if canonical != work_root {
            return Err(SandboxError::Executor(format!(
                "dedicated verifier work_root must already be canonical: configured {}, resolved {}",
                work_root.display(),
                canonical.display()
            )));
        }
        let metadata = fs::symlink_metadata(&canonical).map_err(|source| {
            SandboxError::Executor(format!(
                "could not inspect dedicated verifier work_root {}: {source}",
                canonical.display()
            ))
        })?;
        Ok((canonical, metadata))
    }

    fn validate_service_directory(
        path: &Path,
        service_uid: u32,
        require_private: bool,
        label: &str,
    ) -> Result<(), VerifierExecutorError> {
        let metadata = fs::symlink_metadata(path).map_err(|source| VerifierExecutorError::Io {
            operation: "inspecting executor service directory",
            source,
        })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || metadata.uid() != service_uid
            || (require_private && metadata.mode() & 0o777 != 0o700)
            || (!require_private && metadata.mode() & 0o027 != 0)
        {
            return Err(VerifierExecutorError::InvalidConfig(format!(
                "executor {label} {} is not a protected service-owned directory",
                path.display()
            )));
        }
        Ok(())
    }

    fn validate_trusted_executor_binary(path: &Path) -> Result<(), VerifierExecutorError> {
        if !is_normal_absolute(path) {
            return Err(VerifierExecutorError::InvalidConfig(
                "apptainer executable must be an absolute normalized path".to_string(),
            ));
        }
        let metadata = fs::symlink_metadata(path).map_err(|source| VerifierExecutorError::Io {
            operation: "inspecting apptainer executable",
            source,
        })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.uid() != 0
            || metadata.mode() & 0o022 != 0
            || metadata.mode() & 0o111 == 0
        {
            return Err(VerifierExecutorError::InvalidConfig(format!(
                "apptainer executable {} is not a root-owned, non-writable executable",
                path.display()
            )));
        }
        Ok(())
    }

    fn request_from_spec(spec: &RunSpec) -> Result<(ExecutorRequest, Vec<File>), SandboxError> {
        if spec.network != NetworkPolicy::None {
            return Err(SandboxError::InvalidSpec(
                "staged verifier backend forbids host networking".to_string(),
            ));
        }
        if spec.command.is_empty() {
            return Err(SandboxError::InvalidSpec(
                "protected verifier command must not be empty".to_string(),
            ));
        }
        if spec.binds.len() > MAX_BINDS {
            return Err(SandboxError::InvalidSpec(format!(
                "protected verifier request exceeds {MAX_BINDS} binds"
            )));
        }
        validate_sandbox_path(&spec.workdir, "workdir")?;
        let output_mapping = validate_protected_output_mapping(spec)?;
        let mut assets = Vec::new();
        let image_fd = open_asset(&spec.image, "sandbox image", &mut assets)?;
        let mut binds = Vec::with_capacity(spec.binds.len());
        for bind in &spec.binds {
            validate_sandbox_path(&bind.dst, "bind destination")?;
            let source = match bind.mode {
                BindMode::ReadOnly => ExecutorBindSource::SealedAsset {
                    fd_index: open_asset(&bind.src, "read-only bind", &mut assets)?,
                },
                BindMode::ReadWrite => {
                    if is_owned_descriptor_path(&bind.src) {
                        return Err(SandboxError::InvalidSpec(
                            "read-write verifier binds cannot be descriptor-backed".to_string(),
                        ));
                    }
                    ExecutorBindSource::FreshScratch
                }
            };
            binds.push(ExecutorBind {
                source,
                dst: bind.dst.clone(),
                mode: bind.mode,
                total_limit: bind.total_limit,
                directories: Vec::new(),
            });
        }
        populate_scratch_directories(spec, &mut binds)?;
        if assets.len() > MAX_ASSETS {
            return Err(SandboxError::InvalidSpec(format!(
                "protected verifier request exceeds {MAX_ASSETS} sealed assets"
            )));
        }
        if spec.command.iter().any(|word| word.contains('\0')) {
            return Err(SandboxError::InvalidSpec(
                "protected verifier argv contains a NUL".to_string(),
            ));
        }
        for (key, value) in &spec.env {
            if key.is_empty() || key.contains('=') || key.contains('\0') || value.contains('\0') {
                return Err(SandboxError::InvalidSpec(
                    "protected verifier argv/environment contains an invalid NUL or key"
                        .to_string(),
                ));
            }
        }
        let protected_output = match (&spec.protected_output, output_mapping) {
            (Some(output), Some(mapping)) => Some(ExecutorProtectedOutput {
                bind_index: mapping.bind_index,
                relative_path: mapping.relative_path,
                sandbox_socket: output.sandbox_socket.clone(),
            }),
            (None, None) => None,
            _ => {
                return Err(SandboxError::InvalidSpec(
                    "protected output mapping was internally inconsistent".to_string(),
                ))
            }
        };
        Ok((
            ExecutorRequest {
                version: EXECUTOR_PROTOCOL_VERSION,
                image_fd,
                command: spec.command.clone(),
                binds,
                workdir: spec.workdir.clone(),
                env: spec.env.clone(),
                gpu: spec.gpu,
                network: spec.network,
                limits: spec.limits.clone(),
                protected_output,
                asset_count: assets.len(),
            },
            assets,
        ))
    }

    fn populate_scratch_directories(
        spec: &RunSpec,
        binds: &mut [ExecutorBind],
    ) -> Result<(), SandboxError> {
        for key in SCRATCH_DIRECTORY_ENV {
            let Some((_, value)) = spec.env.iter().find(|(name, _)| name == key) else {
                continue;
            };
            let directory = Path::new(value);
            if !is_normal_absolute(directory) {
                return Err(SandboxError::InvalidSpec(format!(
                    "protected verifier {key} must be an absolute normalized directory"
                )));
            }
            let mut matches = binds
                .iter()
                .enumerate()
                .filter(|(_, bind)| matches!(&bind.source, ExecutorBindSource::FreshScratch))
                .filter_map(|(index, bind)| {
                    directory
                        .strip_prefix(&bind.dst)
                        .ok()
                        .filter(|relative| !relative.as_os_str().is_empty())
                        .map(|relative| (index, relative.to_path_buf()))
                });
            let Some((index, relative)) = matches.next() else {
                continue;
            };
            if matches.next().is_some()
                || relative
                    .components()
                    .any(|component| !matches!(component, std::path::Component::Normal(_)))
            {
                return Err(SandboxError::InvalidSpec(format!(
                    "protected verifier {key} does not map uniquely beneath fresh scratch"
                )));
            }
            if !binds[index].directories.contains(&relative) {
                binds[index].directories.push(relative);
            }
        }
        Ok(())
    }

    fn validate_sandbox_path(path: &Path, label: &str) -> Result<(), SandboxError> {
        if !is_normal_absolute(path) {
            return Err(SandboxError::InvalidSpec(format!(
                "protected verifier {label} must be absolute without '.' or '..': {}",
                path.display()
            )));
        }
        Ok(())
    }

    fn is_normal_absolute(path: &Path) -> bool {
        path.is_absolute()
            && path.components().all(|component| {
                matches!(
                    component,
                    std::path::Component::RootDir | std::path::Component::Normal(_)
                )
            })
    }

    fn open_asset(path: &Path, label: &str, assets: &mut Vec<File>) -> Result<usize, SandboxError> {
        if !is_owned_descriptor_path(path) {
            return Err(SandboxError::InvalidSpec(format!(
                "protected verifier {label} must be an owner-proc sealed descriptor, got {}",
                path.display()
            )));
        }
        let file = File::open(path).map_err(|source| {
            SandboxError::InvalidSpec(format!(
                "could not open protected verifier {label} {}: {source}",
                path.display()
            ))
        })?;
        validate_sealed_asset(&file).map_err(SandboxError::InvalidSpec)?;
        let index = assets.len();
        assets.push(file);
        Ok(index)
    }

    fn validate_sealed_asset(file: &File) -> Result<(), String> {
        let metadata = file
            .metadata()
            .map_err(|error| format!("could not inspect sealed verifier asset: {error}"))?;
        if !metadata.is_file() {
            return Err("sealed verifier asset is not a regular file".to_string());
        }
        let seals = rustix::fs::fcntl_get_seals(file)
            .map_err(|error| format!("could not read verifier asset seals: {error}"))?;
        if !seals.contains(REQUIRED_SEALS) {
            return Err("verifier asset is missing required kernel seals".to_string());
        }
        Ok(())
    }

    fn send_request(
        stream: &mut UnixStream,
        request: &ExecutorWireRequest,
        assets: &[File],
    ) -> std::io::Result<()> {
        if assets.len() > MAX_ASSETS {
            return Err(std::io::Error::other(
                "executor request exceeds descriptor cap",
            ));
        }
        let payload = serde_json::to_vec(request).map_err(std::io::Error::other)?;
        if payload.len() > MAX_REQUEST_BYTES {
            return Err(std::io::Error::other("executor request exceeds byte cap"));
        }
        let length = u32::try_from(payload.len())
            .map_err(|_| std::io::Error::other("executor request length overflow"))?;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&length.to_be_bytes());
        frame.extend_from_slice(&payload);
        if assets.is_empty() {
            return stream.write_all(&frame);
        }
        let borrowed = assets.iter().map(|file| file.as_fd()).collect::<Vec<_>>();
        let mut cmsg_space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_ASSETS))];
        let mut ancillary = SendAncillaryBuffer::new(&mut cmsg_space);
        if !ancillary.push(SendAncillaryMessage::ScmRights(&borrowed)) {
            return Err(std::io::Error::other(
                "could not encode executor asset descriptors",
            ));
        }
        let sent = sendmsg(
            &*stream,
            &[IoSlice::new(&frame)],
            &mut ancillary,
            SendFlags::empty(),
        )?;
        if sent == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "executor request send made no progress",
            ));
        }
        stream.write_all(&frame[sent..])
    }

    fn receive_request(
        stream: &mut UnixStream,
    ) -> std::io::Result<(ExecutorWireRequest, Vec<OwnedFd>)> {
        let mut frame = vec![0_u8; MAX_REQUEST_BYTES + 4];
        let mut cmsg_space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_ASSETS))];
        let mut ancillary = RecvAncillaryBuffer::new(&mut cmsg_space);
        let received = {
            let mut iov = [IoSliceMut::new(&mut frame)];
            recvmsg(&*stream, &mut iov, &mut ancillary, RecvFlags::CMSG_CLOEXEC)?
        };
        if received.bytes == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "executor client closed before request",
            ));
        }
        if received.flags.contains(ReturnFlags::CTRUNC) {
            return Err(std::io::Error::other(
                "executor request descriptor list was truncated",
            ));
        }
        let mut fds = Vec::new();
        for message in ancillary.drain() {
            if let RecvAncillaryMessage::ScmRights(rights) = message {
                fds.extend(rights);
            }
        }
        let mut filled = received.bytes;
        if filled < 4 {
            stream.read_exact(&mut frame[filled..4])?;
            filled = 4;
        }
        let length = u32::from_be_bytes(frame[..4].try_into().unwrap()) as usize;
        if length > MAX_REQUEST_BYTES {
            return Err(std::io::Error::other("executor request exceeds byte cap"));
        }
        let frame_length = 4 + length;
        if filled > frame_length {
            return Err(std::io::Error::other(
                "executor request contained trailing bytes",
            ));
        }
        stream.read_exact(&mut frame[filled..frame_length])?;
        let request =
            serde_json::from_slice(&frame[4..frame_length]).map_err(std::io::Error::other)?;
        Ok((request, fds))
    }

    fn write_response(stream: &mut UnixStream, response: &ExecutorResponse) -> std::io::Result<()> {
        let payload = serde_json::to_vec(response).map_err(std::io::Error::other)?;
        if payload.len() > MAX_RESPONSE_BYTES {
            return Err(std::io::Error::other("executor response exceeds byte cap"));
        }
        let length = u32::try_from(payload.len())
            .map_err(|_| std::io::Error::other("executor response length overflow"))?;
        stream.write_all(&length.to_be_bytes())?;
        stream.write_all(&payload)
    }

    fn read_response(stream: &mut UnixStream) -> std::io::Result<ExecutorResponse> {
        let mut header = [0_u8; 4];
        stream.read_exact(&mut header)?;
        let length = u32::from_be_bytes(header) as usize;
        if length > MAX_RESPONSE_BYTES {
            return Err(std::io::Error::other("executor response exceeds byte cap"));
        }
        let mut payload = vec![0_u8; length];
        stream.read_exact(&mut payload)?;
        serde_json::from_slice(&payload).map_err(std::io::Error::other)
    }

    fn handle_connection(
        mut stream: UnixStream,
        config: &VerifierExecutorConfig,
        service_uid: u32,
    ) -> std::io::Result<()> {
        stream.set_read_timeout(Some(EXECUTOR_HANDSHAKE_TIMEOUT))?;
        stream.set_write_timeout(Some(EXECUTOR_HANDSHAKE_TIMEOUT))?;
        let response = match authenticate_client(&stream, config.client_uid)
            .and_then(|()| receive_request(&mut stream))
        {
            Ok((ExecutorWireRequest::Run(request), fds)) => {
                let wire_timeout = executor_wire_timeout(request.limits.wall);
                stream.set_read_timeout(Some(wire_timeout))?;
                stream.set_write_timeout(Some(wire_timeout))?;
                execute_request(*request, fds, config, service_uid)
            }
            Ok((ExecutorWireRequest::Preflight { version }, _))
                if version != EXECUTOR_PROTOCOL_VERSION =>
            {
                ExecutorResponse::Error(ExecutorWireError::InvalidSpec(format!(
                    "unsupported verifier executor protocol version {version}"
                )))
            }
            Ok((ExecutorWireRequest::Preflight { .. }, fds)) if !fds.is_empty() => {
                ExecutorResponse::Error(ExecutorWireError::InvalidSpec(
                    "verifier executor preflight must not include descriptors".to_string(),
                ))
            }
            Ok((ExecutorWireRequest::Preflight { .. }, _)) => {
                match service_isolation_evidence(config, service_uid) {
                    Ok(evidence) => ExecutorResponse::Preflight {
                        service_uid,
                        evidence,
                    },
                    Err(error) => ExecutorResponse::Error(ExecutorWireError::Executor(format!(
                        "executor isolation preflight failed: {error}"
                    ))),
                }
            }
            Err(error) => ExecutorResponse::Error(ExecutorWireError::Executor(error.to_string())),
        };
        write_response(&mut stream, &response)
    }

    fn authenticate_client(stream: &UnixStream, client_uid: u32) -> std::io::Result<()> {
        let peer = rustix::net::sockopt::socket_peercred(stream)?;
        if peer.uid.as_raw() != client_uid {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!(
                    "executor rejected peer UID {}; expected {client_uid}",
                    peer.uid.as_raw()
                ),
            ));
        }
        Ok(())
    }

    fn execute_request(
        request: ExecutorRequest,
        fds: Vec<OwnedFd>,
        config: &VerifierExecutorConfig,
        service_uid: u32,
    ) -> ExecutorResponse {
        let result = execute_request_inner(request, fds, &config.work_root, &config.apptainer_bin);
        match result {
            Ok(outcome) => ExecutorResponse::Outcome {
                service_uid,
                outcome,
            },
            Err(error) => ExecutorResponse::Error(sandbox_error_to_wire(error)),
        }
    }

    fn execute_request_inner(
        request: ExecutorRequest,
        fds: Vec<OwnedFd>,
        work_root: &Path,
        apptainer_bin: &Path,
    ) -> Result<RunOutcome, SandboxError> {
        validate_wire_request(&request, fds.len())?;
        let request_root = create_request_root(work_root)
            .map_err(|error| SandboxError::Executor(error.to_string()))?;
        let result = stage_and_run(&request_root, request, fds, apptainer_bin);
        let cleanup = remove_request_tree(&request_root);
        match (result, cleanup) {
            (Ok(outcome), Ok(())) => Ok(outcome),
            (Err(error), Ok(())) => Err(error),
            (_, Err(error)) => Err(SandboxError::Executor(format!(
                "could not remove protected request directory {}: {error}",
                request_root.display()
            ))),
        }
    }

    fn open_cleanup_directory<Fd: std::os::fd::AsFd>(
        parent: Fd,
        name: &std::ffi::CStr,
    ) -> std::io::Result<OwnedFd> {
        rustix::fs::chmodat(
            &parent,
            name,
            rustix::fs::Mode::from_bits_truncate(0o700),
            rustix::fs::AtFlags::empty(),
        )?;
        Ok(rustix::fs::openat(
            parent,
            name,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?)
    }

    fn first_cleanup_entry(directory: &OwnedFd) -> std::io::Result<Option<std::ffi::CString>> {
        let scan = rustix::fs::openat(
            directory,
            rustix::cstr!("."),
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?;
        let mut buffer = [MaybeUninit::<u8>::uninit(); 8192];
        let mut entries = rustix::fs::RawDir::new(scan, &mut buffer);
        while let Some(entry) = entries.next() {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_bytes() != b"." && name.to_bytes() != b".." {
                return Ok(Some(name.to_owned()));
            }
        }
        Ok(None)
    }

    fn remove_request_tree(path: &Path) -> std::io::Result<()> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return fs::remove_file(path);
        }

        // The sandbox process tree has been reaped before this runs, so names can
        // no longer race. Keep one anchor plus one scan/child descriptor: return
        // through `..` instead of retaining a descriptor for every depth level.
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
        let mut directory = rustix::fs::openat(
            rustix::fs::CWD,
            path,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?;
        let mut ancestors = Vec::new();
        loop {
            if let Some(name) = first_cleanup_entry(&directory)? {
                let metadata =
                    rustix::fs::statat(&directory, &name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW)?;
                if rustix::fs::FileType::from_raw_mode(metadata.st_mode).is_dir() {
                    let child = open_cleanup_directory(&directory, &name)?;
                    ancestors.push(name);
                    directory = child;
                } else {
                    rustix::fs::unlinkat(&directory, &name, rustix::fs::AtFlags::empty())?;
                }
                continue;
            }

            let Some(name) = ancestors.pop() else {
                drop(directory);
                return fs::remove_dir(path);
            };
            let parent = rustix::fs::openat(
                &directory,
                rustix::cstr!(".."),
                rustix::fs::OFlags::RDONLY
                    | rustix::fs::OFlags::DIRECTORY
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::CLOEXEC,
                rustix::fs::Mode::empty(),
            )?;
            drop(directory);
            rustix::fs::unlinkat(&parent, &name, rustix::fs::AtFlags::REMOVEDIR)?;
            directory = parent;
        }
    }

    fn validate_wire_request(
        request: &ExecutorRequest,
        received_fds: usize,
    ) -> Result<(), SandboxError> {
        if request.version != EXECUTOR_PROTOCOL_VERSION {
            return Err(SandboxError::InvalidSpec(format!(
                "unsupported verifier executor protocol version {}",
                request.version
            )));
        }
        if request.network != NetworkPolicy::None
            || request.asset_count == 0
            || request.asset_count > MAX_ASSETS
            || received_fds != request.asset_count
            || request.image_fd >= request.asset_count
            || request.binds.len() > MAX_BINDS
            || request.command.is_empty()
        {
            return Err(SandboxError::InvalidSpec(
                "malformed staged verifier request".to_string(),
            ));
        }
        validate_sandbox_path(&request.workdir, "workdir")?;
        if request.command.iter().any(|word| word.contains('\0'))
            || request.env.iter().any(|(key, value)| {
                key.is_empty() || key.contains('=') || key.contains('\0') || value.contains('\0')
            })
        {
            return Err(SandboxError::InvalidSpec(
                "executor argv/environment contains an invalid NUL or key".to_string(),
            ));
        }
        let mut referenced = vec![false; request.asset_count];
        referenced[request.image_fd] = true;
        for bind in &request.binds {
            validate_sandbox_path(&bind.dst, "bind destination")?;
            match (&bind.source, bind.mode) {
                (ExecutorBindSource::SealedAsset { fd_index }, BindMode::ReadOnly)
                    if *fd_index < request.asset_count && bind.directories.is_empty() =>
                {
                    referenced[*fd_index] = true;
                }
                (ExecutorBindSource::FreshScratch, BindMode::ReadWrite)
                    if bind.directories.iter().all(|directory| {
                        !directory.as_os_str().is_empty()
                            && !directory.is_absolute()
                            && directory.components().all(|component| {
                                matches!(component, std::path::Component::Normal(_))
                            })
                    }) => {}
                _ => {
                    return Err(SandboxError::InvalidSpec(
                        "executor bind source/mode mismatch".to_string(),
                    ))
                }
            }
        }
        if referenced.iter().any(|value| !value) {
            return Err(SandboxError::InvalidSpec(
                "executor request included an unreferenced descriptor".to_string(),
            ));
        }
        if let Some(output) = &request.protected_output {
            if output.bind_index >= request.binds.len()
                || !matches!(
                    &request.binds[output.bind_index].source,
                    ExecutorBindSource::FreshScratch
                )
                || output.relative_path.as_os_str().is_empty()
                || output.relative_path.is_absolute()
                || output
                    .relative_path
                    .components()
                    .any(|component| !matches!(component, std::path::Component::Normal(_)))
                || request.binds[output.bind_index]
                    .dst
                    .join(&output.relative_path)
                    != output.sandbox_socket
            {
                return Err(SandboxError::InvalidSpec(
                    "executor protected-output mapping is invalid".to_string(),
                ));
            }
            validate_sandbox_path(&output.sandbox_socket, "protected output socket")?;
        }
        Ok(())
    }

    fn create_request_root(work_root: &Path) -> std::io::Result<PathBuf> {
        for _ in 0..1024 {
            let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = work_root.join(format!("request-{}-{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => {
                    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
                    return Ok(path);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::other(
            "could not allocate a unique executor request directory",
        ))
    }

    fn stage_and_run(
        request_root: &Path,
        request: ExecutorRequest,
        fds: Vec<OwnedFd>,
        apptainer_bin: &Path,
    ) -> Result<RunOutcome, SandboxError> {
        let assets_root = request_root.join("assets");
        fs::create_dir(&assets_root).map_err(|error| SandboxError::Executor(error.to_string()))?;
        fs::set_permissions(&assets_root, fs::Permissions::from_mode(0o700))
            .map_err(|error| SandboxError::Executor(error.to_string()))?;
        let mut staged_assets = Vec::with_capacity(fds.len());
        for (index, fd) in fds.into_iter().enumerate() {
            let name = if index == request.image_fd {
                "image.sif".to_string()
            } else {
                format!("asset-{index}")
            };
            let path = assets_root.join(name);
            stage_asset(File::from(fd), &path)?;
            staged_assets.push(path);
        }
        fs::set_permissions(&assets_root, fs::Permissions::from_mode(0o500))
            .map_err(|error| SandboxError::Executor(error.to_string()))?;

        let mut binds = Vec::with_capacity(request.binds.len());
        for (index, bind) in request.binds.into_iter().enumerate() {
            let src = match bind.source {
                ExecutorBindSource::SealedAsset { fd_index } => staged_assets[fd_index].clone(),
                ExecutorBindSource::FreshScratch => {
                    let scratch = request_root.join(format!("scratch-{index}"));
                    fs::create_dir(&scratch)
                        .map_err(|error| SandboxError::Executor(error.to_string()))?;
                    fs::set_permissions(&scratch, fs::Permissions::from_mode(0o700))
                        .map_err(|error| SandboxError::Executor(error.to_string()))?;
                    for directory in &bind.directories {
                        create_private_directory(&scratch, directory)?;
                    }
                    scratch
                }
            };
            binds.push(Bind {
                src,
                dst: bind.dst,
                mode: bind.mode,
                total_limit: bind.total_limit,
            });
        }
        let protected_output = request.protected_output.map(|output| {
            let host_socket = binds[output.bind_index].src.join(output.relative_path);
            ProtectedOutput::new(host_socket, output.sandbox_socket)
        });
        if let Some(output) = &protected_output {
            let parent = output.host_socket.parent().ok_or_else(|| {
                SandboxError::InvalidSpec(
                    "protected output socket has no service-private parent".to_string(),
                )
            })?;
            fs::create_dir_all(parent)
                .map_err(|error| SandboxError::Executor(error.to_string()))?;
            fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
                .map_err(|error| SandboxError::Executor(error.to_string()))?;
        }
        let spec = RunSpec {
            image: staged_assets[request.image_fd].clone(),
            command: request.command,
            binds,
            workdir: request.workdir,
            env: request.env,
            gpu: request.gpu,
            network: request.network,
            limits: request.limits,
            protected_output,
        };
        ApptainerSandbox::with_bin(apptainer_bin).run(&spec)
    }

    fn create_private_directory(root: &Path, relative: &Path) -> Result<(), SandboxError> {
        let mut directory = root.to_path_buf();
        for component in relative.components() {
            let std::path::Component::Normal(component) = component else {
                return Err(SandboxError::InvalidSpec(
                    "fresh scratch directory is not normalized".to_string(),
                ));
            };
            directory.push(component);
            match fs::create_dir(&directory) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let metadata = fs::symlink_metadata(&directory)
                        .map_err(|error| SandboxError::Executor(error.to_string()))?;
                    if metadata.file_type().is_symlink() || !metadata.is_dir() {
                        return Err(SandboxError::Executor(format!(
                            "fresh scratch path is not a directory: {}",
                            directory.display()
                        )));
                    }
                }
                Err(error) => return Err(SandboxError::Executor(error.to_string())),
            }
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))
                .map_err(|error| SandboxError::Executor(error.to_string()))?;
        }
        Ok(())
    }

    fn stage_asset(mut source: File, destination: &Path) -> Result<(), SandboxError> {
        validate_sealed_asset(&source).map_err(SandboxError::InvalidSpec)?;
        source
            .seek(SeekFrom::Start(0))
            .map_err(|error| SandboxError::Executor(error.to_string()))?;
        let mut destination_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o400)
            .open(destination)
            .map_err(|error| SandboxError::Executor(error.to_string()))?;
        let mut source_hash = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let count = source
                .read(&mut buffer)
                .map_err(|error| SandboxError::Executor(error.to_string()))?;
            if count == 0 {
                break;
            }
            source_hash.update(&buffer[..count]);
            destination_file
                .write_all(&buffer[..count])
                .map_err(|error| SandboxError::Executor(error.to_string()))?;
        }
        destination_file
            .sync_all()
            .map_err(|error| SandboxError::Executor(error.to_string()))?;
        drop(destination_file);
        fs::set_permissions(destination, fs::Permissions::from_mode(0o400))
            .map_err(|error| SandboxError::Executor(error.to_string()))?;

        let mut staged =
            File::open(destination).map_err(|error| SandboxError::Executor(error.to_string()))?;
        let mut staged_hash = Sha256::new();
        loop {
            let count = staged
                .read(&mut buffer)
                .map_err(|error| SandboxError::Executor(error.to_string()))?;
            if count == 0 {
                break;
            }
            staged_hash.update(&buffer[..count]);
        }
        if source_hash.finalize() != staged_hash.finalize() {
            return Err(SandboxError::Executor(
                "service-private verifier asset failed copy authentication".to_string(),
            ));
        }
        Ok(())
    }

    fn sandbox_error_to_wire(error: SandboxError) -> ExecutorWireError {
        match error {
            SandboxError::InvalidSpec(message) => ExecutorWireError::InvalidSpec(message),
            SandboxError::Infrastructure { status, stderr } => {
                ExecutorWireError::Infrastructure { status, stderr }
            }
            other => ExecutorWireError::Executor(other.to_string()),
        }
    }

    fn wire_error_to_sandbox(error: ExecutorWireError) -> SandboxError {
        match error {
            ExecutorWireError::InvalidSpec(message) => SandboxError::InvalidSpec(message),
            ExecutorWireError::Infrastructure { status, stderr } => {
                SandboxError::Infrastructure { status, stderr }
            }
            ExecutorWireError::Executor(message) => SandboxError::Executor(message),
        }
    }

    #[cfg(test)]
    mod tests {
        use std::os::fd::AsRawFd as _;
        use std::os::unix::fs::symlink;
        use std::time::Duration;

        use super::*;

        const ENTRY_MARKER: &str = "ferrl-sandbox-verifier-entry-v1";

        fn sealed_asset(bytes: &[u8]) -> File {
            let descriptor = rustix::fs::memfd_create(
                "ferrl-verifier-executor-test",
                rustix::fs::MemfdFlags::ALLOW_SEALING | rustix::fs::MemfdFlags::CLOEXEC,
            )
            .unwrap();
            let mut file = File::from(descriptor);
            file.write_all(bytes).unwrap();
            rustix::fs::fcntl_add_seals(&file, REQUIRED_SEALS).unwrap();
            file
        }

        fn descriptor_path(file: &File) -> PathBuf {
            PathBuf::from(format!(
                "/proc/{}/fd/{}",
                std::process::id(),
                file.as_raw_fd()
            ))
        }

        fn test_root(label: &str) -> PathBuf {
            let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "ferrl-verifier-executor-{label}-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
            root
        }

        fn fake_apptainer(root: &Path, body: &str) -> PathBuf {
            let path = root.join("fake-apptainer");
            fs::write(&path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            path
        }

        fn other_nonroot_uid() -> u32 {
            if rustix::process::geteuid().as_raw() == 1 {
                2
            } else {
                1
            }
        }

        fn dedicated_evidence_fixture(service_uid: u32) -> VerifierIsolationEvidence {
            VerifierIsolationEvidence {
                contract_version: VERIFIER_ISOLATION_EVIDENCE_VERSION,
                tier: VerifierIsolationTier::DedicatedUidServiceV1,
                requester_uid: rustix::process::geteuid().as_raw(),
                launcher_uid: service_uid,
                uid_boundary: VerifierUidBoundary::DistinctHostUid,
                asset_transport: VerifierAssetTransport::ScmRightsSealedCopy,
                apptainer_path: PathBuf::from("/usr/bin/apptainer"),
                apptainer_sha256: "a".repeat(64),
                apptainer_len_bytes: 1,
                apptainer_version: "apptainer version 1".to_string(),
                work_root: PathBuf::from("/var/lib/ferrl/verifier"),
                work_root_uid: service_uid,
                work_root_device: 1,
                work_root_inode: 1,
                work_root_mode: 0o700,
            }
        }

        fn complete_request(spec: &RunSpec, config: &VerifierExecutorConfig) -> ExecutorResponse {
            let (request, assets) = request_from_spec(spec).unwrap();
            let (mut client, server) = UnixStream::pair().unwrap();
            let config = config.clone();
            let service_uid = rustix::process::geteuid().as_raw();
            let executor = thread::spawn(move || handle_connection(server, &config, service_uid));
            send_request(
                &mut client,
                &ExecutorWireRequest::Run(Box::new(request)),
                &assets,
            )
            .unwrap();
            let response = read_response(&mut client).unwrap();
            executor.join().unwrap().unwrap();
            response
        }

        fn scratch_probe_script(action: &str) -> String {
            format!(
                r#"scratch=''
previous=''
for argument in "$@"; do
    if [ "$previous" = '--bind' ]; then
        case "$argument" in
            *:/work:rw) scratch=$(expr "$argument" : '\(.*\):/work:rw') ;;
        esac
    fi
    previous=$argument
done
[ -n "$scratch" ]
[ -d "$scratch/cache/triton" ]
printf '%s\n' '{ENTRY_MARKER}' >&2
{action}"#
            )
        }

        #[test]
        #[allow(clippy::cognitive_complexity)]
        fn verifier_backends_report_explicit_versioned_tiers() {
            let same_uid = SameUidApptainerSandbox::new("/private/work")
                .with_apptainer_bin("/usr/bin/apptainer");
            assert_eq!(
                same_uid.isolation_tier(),
                VerifierIsolationTier::SameUidApptainerV1
            );
            assert_eq!(same_uid.work_root(), Path::new("/private/work"));
            assert_eq!(same_uid.apptainer_bin(), Path::new("/usr/bin/apptainer"));
            assert_eq!(
                serde_json::to_string(&same_uid.isolation_tier()).unwrap(),
                "\"same_uid_apptainer_v1\""
            );
            assert_eq!(same_uid.isolation_tier().as_str(), "same_uid_apptainer_v1");

            let dedicated = VerifierExecutorSandbox::new("/run/ferrl/executor.sock");
            assert_eq!(
                dedicated.isolation_tier(),
                VerifierIsolationTier::DedicatedUidServiceV1
            );
            assert_eq!(
                serde_json::to_string(&dedicated.isolation_tier()).unwrap(),
                "\"dedicated_uid_service_v1\""
            );

            let default = SameUidApptainerSandbox::default();
            assert_eq!(default.apptainer_bin(), Path::new("/usr/bin/apptainer"));
            assert!(default
                .work_root()
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|name| name.starts_with(SAME_UID_WORK_ROOT_PREFIX)));
        }

        #[test]
        #[allow(clippy::cognitive_complexity)]
        fn same_uid_backend_rejects_invalid_local_configuration_without_fallback() {
            let relative = SameUidApptainerSandbox::new("relative/work")
                .with_apptainer_bin("/usr/bin/apptainer");
            let error = relative
                .run(&RunSpec::new("/unused/image", vec!["true".into()]))
                .unwrap_err()
                .to_string();
            assert!(error.contains("same-UID verifier"), "{error}");
            assert!(
                error.contains("work_root") || error.contains("non-root"),
                "{error}"
            );
            assert!(
                !error.contains("socket"),
                "same-UID backend fell through: {error}"
            );

            let root = test_root("same-uid-config");
            let backend = SameUidApptainerSandbox::new(&root).with_apptainer_bin("/usr/bin/true");
            if rustix::process::geteuid().as_raw() == 0 {
                let error = backend.preflight().unwrap_err().to_string();
                assert!(error.contains("non-root UID"), "{error}");
            } else {
                let evidence = backend.preflight().unwrap();
                assert_eq!(
                    evidence.contract_version,
                    VERIFIER_ISOLATION_EVIDENCE_VERSION
                );
                assert_eq!(evidence.tier, VerifierIsolationTier::SameUidApptainerV1);
                assert_eq!(evidence.requester_uid, rustix::process::geteuid().as_raw());
                assert_eq!(evidence.launcher_uid, evidence.requester_uid);
                assert_eq!(evidence.uid_boundary, VerifierUidBoundary::SameHostUid);
                assert_eq!(
                    evidence.asset_transport,
                    VerifierAssetTransport::InProcessSealedCopy
                );
                assert_eq!(evidence.apptainer_path, Path::new("/usr/bin/true"));
                assert_eq!(evidence.apptainer_sha256.len(), 64);
                assert!(evidence.apptainer_len_bytes > 0);
                assert!(!evidence.apptainer_version.is_empty());
                assert_eq!(evidence.work_root, root);
                assert_eq!(evidence.work_root_uid, evidence.launcher_uid);
                assert_ne!(evidence.work_root_inode, 0);
                assert_eq!(evidence.work_root_mode, 0o700);
                let round_trip: VerifierIsolationEvidence =
                    serde_json::from_slice(&serde_json::to_vec(&evidence).unwrap()).unwrap();
                assert_eq!(round_trip, evidence);
                fs::set_permissions(&root, fs::Permissions::from_mode(0o770)).unwrap();
                let error = backend.preflight().unwrap_err().to_string();
                assert!(error.contains("mode 0700"), "{error}");
                fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();

                let untrusted = fake_apptainer(&root, "exit 0");
                fs::set_permissions(&untrusted, fs::Permissions::from_mode(0o770)).unwrap();
                let error = validate_same_uid_config(
                    &SameUidApptainerSandbox::new(&root).with_apptainer_bin(untrusted),
                )
                .unwrap_err()
                .to_string();
                assert!(error.contains("not trusted"), "{error}");
            }
            remove_request_tree(&root).unwrap();
        }

        #[test]
        #[allow(clippy::cognitive_complexity)]
        fn same_uid_preflight_creates_a_private_canonical_work_root() {
            if rustix::process::geteuid().as_raw() == 0 {
                return;
            }
            let parent = test_root("same-uid-create-root");
            let work_root = parent.join("work");
            let evidence = SameUidApptainerSandbox::new(&work_root)
                .with_apptainer_bin("/usr/bin/true")
                .preflight()
                .unwrap();
            let metadata = fs::symlink_metadata(&work_root).unwrap();
            assert!(metadata.is_dir());
            assert_eq!(metadata.uid(), rustix::process::geteuid().as_raw());
            assert_eq!(metadata.mode() & 0o777, 0o700);
            assert_eq!(evidence.work_root, work_root);
            assert_eq!(evidence.work_root_device, metadata.dev());
            assert_eq!(evidence.work_root_inode, metadata.ino());
            remove_request_tree(&parent).unwrap();
        }

        #[test]
        fn same_uid_path_reuses_sealed_staging_fresh_scratch_and_cleanup() {
            let root = test_root("same-uid-staging");
            let work_root = root.join("work");
            fs::create_dir(&work_root).unwrap();
            fs::set_permissions(&work_root, fs::Permissions::from_mode(0o700)).unwrap();
            let discarded_client_scratch = root.join("client-scratch");
            fs::create_dir(&discarded_client_scratch).unwrap();
            fs::write(discarded_client_scratch.join("keep"), b"client-owned").unwrap();
            let executable = fake_apptainer(&root, &scratch_probe_script("exit 31"));
            let image = sealed_asset(b"image");
            let spec = RunSpec::new(descriptor_path(&image), vec!["true".into()])
                .with_binds(vec![Bind::rw(&discarded_client_scratch, "/work")])
                .with_env(vec![
                    ("HOME".into(), "/work/cache".into()),
                    ("TRITON_CACHE_DIR".into(), "/work/cache/triton".into()),
                ]);
            let (request, assets) = request_from_spec(&spec).unwrap();
            let fds = assets.into_iter().map(OwnedFd::from).collect();

            let outcome = execute_request_inner(request, fds, &work_root, &executable).unwrap();
            assert_eq!(outcome.status, RunStatus::Exited(31));
            assert_eq!(
                fs::read(discarded_client_scratch.join("keep")).unwrap(),
                b"client-owned"
            );
            assert!(fs::read_dir(&work_root).unwrap().next().is_none());
            remove_request_tree(&root).unwrap();
        }

        #[test]
        #[allow(clippy::cognitive_complexity)] // one request asserts every path-free source invariant
        fn request_requires_only_sealed_assets_and_fresh_scratch() {
            let image = sealed_asset(b"image");
            let eval = sealed_asset(b"eval");
            let spec = RunSpec::new(descriptor_path(&image), vec!["true".into()])
                .with_binds(vec![
                    Bind::ro(descriptor_path(&eval), "/opt/eval.py"),
                    Bind::rw("/untrusted/client/scratch", "/work"),
                ])
                .with_env(vec![
                    ("HOME".into(), "/work/cache".into()),
                    ("TRITON_CACHE_DIR".into(), "/work/cache/triton".into()),
                ])
                .with_protected_output(ProtectedOutput::new(
                    "/untrusted/client/scratch/grade.sock",
                    "/work/grade.sock",
                ));
            let (request, assets) = request_from_spec(&spec).unwrap();
            assert_eq!(assets.len(), 2);
            assert_eq!(request.image_fd, 0);
            assert!(matches!(
                request.binds[0].source,
                ExecutorBindSource::SealedAsset { fd_index: 1 }
            ));
            assert!(matches!(
                request.binds[1].source,
                ExecutorBindSource::FreshScratch
            ));
            assert_eq!(
                request.binds[1].directories,
                [PathBuf::from("cache"), PathBuf::from("cache/triton")]
            );
            assert_eq!(
                request.protected_output.unwrap().relative_path,
                PathBuf::from("grade.sock")
            );

            let ordinary = RunSpec::new("/tmp/image.sif", vec!["true".into()]);
            assert!(request_from_spec(&ordinary).is_err());
        }

        #[test]
        fn scm_rights_request_round_trip_preserves_sealed_descriptors() {
            let image = sealed_asset(b"image");
            let spec = RunSpec::new(descriptor_path(&image), vec!["true".into()]);
            let (request, assets) = request_from_spec(&spec).unwrap();
            let (mut sender, mut receiver) = UnixStream::pair().unwrap();
            let writer = thread::spawn(move || {
                send_request(
                    &mut sender,
                    &ExecutorWireRequest::Run(Box::new(request)),
                    &assets,
                )
            });
            let (received, fds) = receive_request(&mut receiver).unwrap();
            writer.join().unwrap().unwrap();
            let ExecutorWireRequest::Run(received) = received else {
                panic!("run request became a preflight request");
            };
            assert_eq!(received.asset_count, 1);
            assert_eq!(fds.len(), 1);
            validate_sealed_asset(&File::from(fds.into_iter().next().unwrap())).unwrap();
        }

        #[test]
        fn preflight_request_round_trip_requires_no_descriptors() {
            let (mut sender, mut receiver) = UnixStream::pair().unwrap();
            let writer = thread::spawn(move || {
                send_request(
                    &mut sender,
                    &ExecutorWireRequest::Preflight {
                        version: EXECUTOR_PROTOCOL_VERSION,
                    },
                    &[],
                )
            });
            let (received, fds) = receive_request(&mut receiver).unwrap();
            writer.join().unwrap().unwrap();
            assert!(matches!(
                received,
                ExecutorWireRequest::Preflight {
                    version: EXECUTOR_PROTOCOL_VERSION
                }
            ));
            assert!(fds.is_empty());
        }

        #[test]
        fn dedicated_evidence_validation_is_tier_and_peer_bound() {
            let service_uid = other_nonroot_uid();
            let evidence = dedicated_evidence_fixture(service_uid);
            validate_dedicated_evidence(&evidence, service_uid).unwrap();

            let error = validate_dedicated_evidence(&evidence, rustix::process::geteuid().as_raw())
                .unwrap_err()
                .to_string();
            assert!(error.contains("invalid dedicated-tier"), "{error}");

            let mut mislabeled = evidence;
            mislabeled.uid_boundary = VerifierUidBoundary::SameHostUid;
            let error = validate_dedicated_evidence(&mislabeled, service_uid)
                .unwrap_err()
                .to_string();
            assert!(error.contains("invalid dedicated-tier"), "{error}");
        }

        #[test]
        fn dedicated_preflight_response_round_trip_preserves_evidence() {
            let service_uid = other_nonroot_uid();
            let evidence = dedicated_evidence_fixture(service_uid);
            let response = ExecutorResponse::Preflight {
                service_uid,
                evidence: evidence.clone(),
            };
            let (mut sender, mut receiver) = UnixStream::pair().unwrap();
            let writer = thread::spawn(move || write_response(&mut sender, &response));
            let received = read_response(&mut receiver).unwrap();
            writer.join().unwrap().unwrap();
            let ExecutorResponse::Preflight {
                service_uid: received_uid,
                evidence: received_evidence,
            } = received
            else {
                panic!("dedicated preflight response changed kind");
            };
            assert_eq!(received_uid, service_uid);
            assert_eq!(received_evidence, evidence);
            validate_dedicated_evidence(&received_evidence, received_uid).unwrap();
        }

        #[test]
        fn authenticated_preflight_cannot_self_label_a_same_uid_service() {
            let root = test_root("preflight-same-uid");
            let current_uid = rustix::process::geteuid().as_raw();
            let config =
                VerifierExecutorConfig::new(root.join("executor.sock"), &root, current_uid)
                    .with_apptainer_bin("/usr/bin/true");
            let (mut client, server) = UnixStream::pair().unwrap();
            let executor = thread::spawn(move || handle_connection(server, &config, current_uid));
            send_request(
                &mut client,
                &ExecutorWireRequest::Preflight {
                    version: EXECUTOR_PROTOCOL_VERSION,
                },
                &[],
            )
            .unwrap();
            let response = read_response(&mut client).unwrap();
            executor.join().unwrap().unwrap();
            let ExecutorResponse::Error(ExecutorWireError::Executor(error)) = response else {
                panic!("same-UID service produced dedicated-tier evidence");
            };
            assert!(error.contains("distinct observed non-root"), "{error}");
            remove_request_tree(&root).unwrap();
        }

        #[test]
        fn complete_request_returns_original_outcome_and_removes_staging_tree() {
            let root = test_root("complete-request");
            let work_root = root.join("work");
            fs::create_dir(&work_root).unwrap();
            fs::set_permissions(&work_root, fs::Permissions::from_mode(0o700)).unwrap();
            let executable = fake_apptainer(&root, &scratch_probe_script("exit 23"));
            let config = VerifierExecutorConfig::new(
                root.join("executor.sock"),
                &work_root,
                rustix::process::geteuid().as_raw(),
            )
            .with_apptainer_bin(executable);
            let image = sealed_asset(b"image");
            let spec = RunSpec::new(descriptor_path(&image), vec!["true".into()])
                .with_binds(vec![Bind::rw("/discarded/client/scratch", "/work")])
                .with_env(vec![
                    ("HOME".into(), "/work/cache".into()),
                    ("TRITON_CACHE_DIR".into(), "/work/cache/triton".into()),
                ]);

            let response = complete_request(&spec, &config);
            assert!(matches!(
                response,
                ExecutorResponse::Outcome {
                    outcome: RunOutcome {
                        status: RunStatus::Exited(23),
                        ..
                    },
                    ..
                }
            ));
            assert!(fs::read_dir(&work_root).unwrap().next().is_none());
            remove_request_tree(&root).unwrap();
        }

        #[test]
        fn candidate_permission_changes_cannot_replace_the_sandbox_outcome() {
            for (label, action) in [
                ("root-mode-zero", "chmod 000 \"$scratch\"; exit 29"),
                (
                    "nested-mode-zero",
                    "mkdir \"$scratch/locked\"; chmod 000 \"$scratch/locked\"; exit 29",
                ),
            ] {
                let root = test_root(label);
                let work_root = root.join("work");
                fs::create_dir(&work_root).unwrap();
                fs::set_permissions(&work_root, fs::Permissions::from_mode(0o700)).unwrap();
                let executable = fake_apptainer(&root, &scratch_probe_script(action));
                let config = VerifierExecutorConfig::new(
                    root.join("executor.sock"),
                    &work_root,
                    rustix::process::geteuid().as_raw(),
                )
                .with_apptainer_bin(executable);
                let image = sealed_asset(b"image");
                let spec = RunSpec::new(descriptor_path(&image), vec!["true".into()])
                    .with_binds(vec![
                        Bind::rw("/discarded/client/scratch", "/work").with_total_limit(1 << 20)
                    ])
                    .with_env(vec![
                        ("HOME".into(), "/work/cache".into()),
                        ("TRITON_CACHE_DIR".into(), "/work/cache/triton".into()),
                    ]);

                let response = complete_request(&spec, &config);
                let ExecutorResponse::Outcome { outcome, .. } = response else {
                    panic!("candidate permission change became an executor error");
                };
                assert!(matches!(
                    outcome.status,
                    RunStatus::ScratchExceeded | RunStatus::Exited(29)
                ));
                assert!(fs::read_dir(&work_root).unwrap().next().is_none());
                remove_request_tree(&root).unwrap();
            }
        }

        #[test]
        fn cleanup_restores_directories_without_following_symlinks() {
            let root = test_root("cleanup-symlink");
            let request = root.join("request");
            let locked = request.join("locked");
            let outside = root.join("outside");
            fs::create_dir(&request).unwrap();
            fs::create_dir(&locked).unwrap();
            fs::create_dir(&outside).unwrap();
            fs::write(outside.join("keep"), b"outside").unwrap();
            symlink(&outside, locked.join("outside-link")).unwrap();
            let mut deep = locked.clone();
            for _ in 0..1024 {
                deep.push("d");
                fs::create_dir(&deep).unwrap();
            }
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
            fs::set_permissions(&request, fs::Permissions::from_mode(0o000)).unwrap();

            remove_request_tree(&request).unwrap();
            assert!(!request.exists());
            assert_eq!(fs::read(outside.join("keep")).unwrap(), b"outside");
            remove_request_tree(&root).unwrap();
        }

        #[test]
        fn cleanup_succeeds_under_constrained_nofile_limit() {
            const CHILD_ENV: &str = "FERRL_TEST_CONSTRAINED_CLEANUP_CHILD";
            if std::env::var_os(CHILD_ENV).is_some() {
                let root = test_root("cleanup-low-nofile");
                let request = root.join("request");
                fs::create_dir(&request).unwrap();
                let mut deep = request.clone();
                for _ in 0..256 {
                    deep.push("d");
                    fs::create_dir(&deep).unwrap();
                }
                fs::set_permissions(&request, fs::Permissions::from_mode(0o000)).unwrap();
                remove_request_tree(&request).unwrap();
                assert!(!request.exists());
                remove_request_tree(&root).unwrap();
                return;
            }

            let executable = std::env::current_exe().unwrap();
            let mut child = std::process::Command::new("/bin/sh");
            child
                .arg("-c")
                .arg("ulimit -n 32\nexec \"$@\"")
                .arg("ferrl-cleanup-test")
                .arg(executable)
                .arg("cleanup_succeeds_under_constrained_nofile_limit")
                .arg("--nocapture")
                .arg("--test-threads=1")
                .env(CHILD_ENV, "1");
            let output = child.output().unwrap();
            assert!(
                output.status.success(),
                "low-RLIMIT cleanup child failed: stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        #[test]
        fn worst_case_escaped_captures_fit_the_response_wire() {
            let capture = "\0".repeat(CAPTURE_CAP);
            let response = ExecutorResponse::Outcome {
                service_uid: 123,
                outcome: RunOutcome {
                    status: RunStatus::Exited(0),
                    stdout: capture.clone(),
                    stderr: capture.clone(),
                    protected_output: capture,
                    wall: Duration::from_secs(1),
                },
            };
            let (mut sender, mut receiver) = UnixStream::pair().unwrap();
            let writer = thread::spawn(move || write_response(&mut sender, &response));
            let received = read_response(&mut receiver).unwrap();
            writer.join().unwrap().unwrap();
            let ExecutorResponse::Outcome {
                service_uid,
                outcome,
            } = received
            else {
                panic!("worst-case captures did not round-trip as an outcome");
            };
            assert_eq!(service_uid, 123);
            assert_eq!(outcome.stdout.len(), CAPTURE_CAP);
            assert_eq!(outcome.stderr.len(), CAPTURE_CAP);
            assert_eq!(outcome.protected_output.len(), CAPTURE_CAP);
        }

        #[test]
        fn infrastructure_error_round_trips_the_response_wire() {
            let response = ExecutorResponse::Error(ExecutorWireError::Infrastructure {
                status: RunStatus::TimedOut,
                stderr: "trusted runtime failed".to_string(),
            });
            let (mut sender, mut receiver) = UnixStream::pair().unwrap();
            let writer = thread::spawn(move || write_response(&mut sender, &response));
            let received = read_response(&mut receiver).unwrap();
            writer.join().unwrap().unwrap();
            let ExecutorResponse::Error(error) = received else {
                panic!("infrastructure error became an executor outcome");
            };
            assert!(matches!(
                wire_error_to_sandbox(error),
                SandboxError::Infrastructure {
                    status: RunStatus::TimedOut,
                    ref stderr,
                } if stderr == "trusted runtime failed"
            ));
        }

        #[test]
        fn executor_deadline_includes_the_run_budget_and_transport_grace() {
            assert_eq!(
                executor_wire_timeout(Duration::from_secs(7)),
                Duration::from_secs(37)
            );
            assert_eq!(
                executor_wire_timeout(Duration::ZERO),
                Duration::from_secs(31)
            );
        }

        #[test]
        fn service_rejects_relative_binary_and_group_writable_socket_parent() {
            let binary_error = validate_trusted_executor_binary(Path::new("apptainer"))
                .unwrap_err()
                .to_string();
            assert!(binary_error.contains("absolute normalized path"));

            let root = test_root("group-writable-parent");
            fs::set_permissions(&root, fs::Permissions::from_mode(0o770)).unwrap();
            let directory_error = validate_service_directory(
                &root,
                rustix::process::geteuid().as_raw(),
                false,
                "socket parent",
            )
            .unwrap_err()
            .to_string();
            assert!(directory_error.contains("protected service-owned directory"));
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
            remove_request_tree(&root).unwrap();
        }

        #[test]
        fn service_private_copy_matches_the_sealed_source() {
            let root = test_root("copy");
            let destination = root.join("asset");
            stage_asset(sealed_asset(b"authenticated bytes"), &destination).unwrap();
            assert_eq!(fs::read(&destination).unwrap(), b"authenticated bytes");
            assert_eq!(fs::metadata(&destination).unwrap().mode() & 0o777, 0o400);
            remove_request_tree(&root).unwrap();
        }

        #[test]
        fn service_rejects_same_uid_deployment() {
            let root = std::env::temp_dir().join("ferrl-executor-same-uid");
            let config = VerifierExecutorConfig::new(
                root.join("executor.sock"),
                root.join("work"),
                rustix::process::geteuid().as_raw(),
            );
            let error = validate_service_config(&config).unwrap_err().to_string();
            assert!(error.contains("distinct non-root UIDs"));
        }
    }
}
