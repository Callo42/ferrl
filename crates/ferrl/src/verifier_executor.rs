//! Protected, dedicated-UID execution for sealed verifier assets.
//!
//! [`VerifierExecutorSandbox`] sends a path-free run request plus fully sealed
//! verifier file descriptors over an authenticated Unix socket. The executor
//! must run as a dedicated non-root UID distinct from the requesting training
//! UID. It copies those descriptors into a service-private request directory,
//! creates fresh writable binds there, and owns the complete Apptainer launch.
//! The candidate therefore never shares a user-namespace capability boundary or
//! a host UID with the training process that owns reward publication.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::sandbox::{
    Bind, BindMode, NetworkPolicy, ProtectedOutput, ResourceLimits, RunOutcome, RunSpec, RunStatus,
    Sandbox, SandboxError,
};

/// Default protected verifier executor socket.
pub const DEFAULT_VERIFIER_EXECUTOR_SOCKET: &str = "/run/ferrl/verifier-executor.sock";

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
    /// Apptainer executable used by the service.
    pub apptainer_bin: PathBuf,
    /// Socket permission bits, normally `0o660` with an administrator-managed
    /// service group shared with the training UID.
    pub socket_mode: u32,
}

impl VerifierExecutorConfig {
    /// Construct a service configuration with `apptainer` from `PATH` and a
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
            apptainer_bin: PathBuf::from("apptainer"),
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
    use std::os::fd::{AsFd as _, AsRawFd as _, OwnedFd};
    use std::os::unix::fs::{
        FileTypeExt as _, MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _,
    };
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

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

    const EXECUTOR_PROTOCOL_VERSION: u32 = 1;
    const MAX_ASSETS: usize = 32;
    const MAX_BINDS: usize = 32;
    const MAX_REQUEST_BYTES: usize = 1024 * 1024;
    const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
    const JSON_CONTROL_ESCAPE_BYTES: usize = 6;
    const OUTCOME_CAPTURE_FIELDS: usize = 3;
    const RESPONSE_OVERHEAD_BYTES: usize = 1024 * 1024;
    const _: () = assert!(
        CAPTURE_CAP * JSON_CONTROL_ESCAPE_BYTES * OUTCOME_CAPTURE_FIELDS + RESPONSE_OVERHEAD_BYTES
            <= MAX_RESPONSE_BYTES
    );
    const SCRATCH_DIRECTORY_ENV: [&str; 2] = ["HOME", "TRITON_CACHE_DIR"];
    const REQUIRED_SEALS: rustix::fs::SealFlags = rustix::fs::SealFlags::WRITE
        .union(rustix::fs::SealFlags::GROW)
        .union(rustix::fs::SealFlags::SHRINK)
        .union(rustix::fs::SealFlags::SEAL);
    static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    pub(super) fn run_client(
        socket_path: &Path,
        spec: &RunSpec,
    ) -> Result<RunOutcome, SandboxError> {
        let service_uid = validate_client_socket(socket_path)?;
        let (request, assets) = request_from_spec(spec)?;
        let mut stream = UnixStream::connect(socket_path).map_err(|source| {
            SandboxError::Executor(format!(
                "could not connect to {}: {source}",
                socket_path.display()
            ))
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
        send_request(&mut stream, &request, &assets)
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
            } => Err(SandboxError::Executor(format!(
                "executor response UID {response_uid} does not match authenticated UID {service_uid}"
            ))),
            ExecutorResponse::Error(error) => Err(wire_error_to_sandbox(error)),
        }
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
        if metadata.uid() == client_uid {
            return Err(SandboxError::Executor(
                "executor socket is owned by the training UID; a dedicated service UID is required"
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
            || parent_metadata.mode() & 0o007 != 0
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
        if service_uid == 0 || service_uid == config.client_uid {
            return Err(VerifierExecutorError::InvalidConfig(
                "executor must run under a dedicated non-root UID distinct from client_uid"
                    .to_string(),
            ));
        }
        if !is_normal_absolute(&config.socket_path) || !is_normal_absolute(&config.work_root) {
            return Err(VerifierExecutorError::InvalidConfig(
                "socket_path and work_root must be absolute without '.' or '..'".to_string(),
            ));
        }
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
        Ok(service_uid)
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
            || (!require_private && metadata.mode() & 0o007 != 0)
        {
            return Err(VerifierExecutorError::InvalidConfig(format!(
                "executor {label} {} is not a protected service-owned directory",
                path.display()
            )));
        }
        Ok(())
    }

    fn request_from_spec(spec: &RunSpec) -> Result<(ExecutorRequest, Vec<File>), SandboxError> {
        if spec.network != NetworkPolicy::None {
            return Err(SandboxError::InvalidSpec(
                "protected verifier executor forbids host networking".to_string(),
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
        request: &ExecutorRequest,
        assets: &[File],
    ) -> std::io::Result<()> {
        let payload = serde_json::to_vec(request).map_err(std::io::Error::other)?;
        if payload.len() > MAX_REQUEST_BYTES {
            return Err(std::io::Error::other("executor request exceeds byte cap"));
        }
        let length = u32::try_from(payload.len())
            .map_err(|_| std::io::Error::other("executor request length overflow"))?;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&length.to_be_bytes());
        frame.extend_from_slice(&payload);
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
    ) -> std::io::Result<(ExecutorRequest, Vec<OwnedFd>)> {
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
        let response = match authenticate_client(&stream, config.client_uid)
            .and_then(|()| receive_request(&mut stream))
        {
            Ok((request, fds)) => execute_request(request, fds, config, service_uid),
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
        let result = execute_request_inner(request, fds, config);
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
        config: &VerifierExecutorConfig,
    ) -> Result<RunOutcome, SandboxError> {
        validate_wire_request(&request, fds.len())?;
        let request_root = create_request_root(&config.work_root)
            .map_err(|error| SandboxError::Executor(error.to_string()))?;
        let result = stage_and_run(&request_root, request, fds, config);
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

    struct CleanupDirectory {
        path: PathBuf,
        directory: File,
        entries: fs::ReadDir,
    }

    fn open_cleanup_directory(path: PathBuf) -> std::io::Result<CleanupDirectory> {
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
        let directory = File::open(&path)?;
        let descriptor_path = PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd()));
        let entries = fs::read_dir(descriptor_path)?;
        Ok(CleanupDirectory {
            path,
            directory,
            entries,
        })
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

        // The sandbox process tree has been reaped before this runs. Restore only
        // owner access on directory inodes reached without following symlinks.
        // Parent descriptors stay open while children are traversed through
        // /proc/self/fd, so hostile depth cannot exceed PATH_MAX or the Rust stack.
        let mut directories = vec![open_cleanup_directory(path.to_path_buf())?];
        while !directories.is_empty() {
            let entry = directories
                .last_mut()
                .expect("the cleanup stack was checked as nonempty")
                .entries
                .next()
                .transpose()?;
            if let Some(entry) = entry {
                let child = entry.path();
                let metadata = fs::symlink_metadata(&child)?;
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    fs::remove_file(child)?;
                } else {
                    directories.push(open_cleanup_directory(child)?);
                }
                continue;
            }

            let CleanupDirectory {
                path,
                directory,
                entries,
            } = directories
                .pop()
                .expect("the cleanup stack was checked as nonempty");
            drop(entries);
            drop(directory);
            fs::remove_dir(path)?;
        }
        Ok(())
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
                "malformed protected verifier executor request".to_string(),
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
        config: &VerifierExecutorConfig,
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
        ApptainerSandbox::with_bin(&config.apptainer_bin).run(&spec)
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

        fn complete_request(spec: &RunSpec, config: &VerifierExecutorConfig) -> ExecutorResponse {
            let (request, assets) = request_from_spec(spec).unwrap();
            let (mut client, server) = UnixStream::pair().unwrap();
            let config = config.clone();
            let service_uid = rustix::process::geteuid().as_raw();
            let executor = thread::spawn(move || handle_connection(server, &config, service_uid));
            send_request(&mut client, &request, &assets).unwrap();
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
            let writer = thread::spawn(move || send_request(&mut sender, &request, &assets));
            let (received, fds) = receive_request(&mut receiver).unwrap();
            writer.join().unwrap().unwrap();
            assert_eq!(received.asset_count, 1);
            assert_eq!(fds.len(), 1);
            validate_sealed_asset(&File::from(fds.into_iter().next().unwrap())).unwrap();
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
            assert!(error.contains("dedicated non-root UID"));
        }
    }
}
