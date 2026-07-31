"""Ferrl-owned TriMul protected verifier boundary.

The parent owns hidden inputs, checking, statistics, timing, and the grade socket. A
non-dumpable controller owns the trusted status/output connections, and candidate
Python runs in a separate payload process with independent CUDA allocations and only
an untrusted request/result channel. Bounded CPU byte strings cross each process
boundary: no parent CUDA allocation is ever exported through CUDA IPC. The versioned
timing metric covers request handoff, controller/payload scheduling, candidate
execution, device-to-host capture, and result receipt; trusted reconstruction and
checking happen after the timer stops.
"""

import ctypes
import errno
import io
import importlib
import json
import multiprocessing
import os
import signal
import socket
import stat
import sys
import time


PR_GET_DUMPABLE = 3
PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38
PR_GET_NO_NEW_PRIVS = 39
PR_SET_CHILD_SUBREAPER = 36
SUBMISSION_PATH = "/opt/ferrl-verifier/submission.py"
RESULT_SPLIT = "===FERRL-BENCH==="
ISOLATION_TIMING_METRICS = {
    "same_uid_apptainer_v1": "same-uid-apptainer-latency-v1",
    "dedicated_uid_service_v1": "isolated-service-latency-v1",
}
MAX_STATUS_BYTES = 1024
MAX_STATUS_EVENTS = 32
MAX_CASE_SEED = (1 << 32) - 1
ENTRY_ACK = b"ENTRY-ACK-v3"
# The pinned task's largest float32 input is 3 GiB before its small mask,
# weights, config, and serialization envelope. Keep the transport bounded while
# admitting that complete launch-bound case.
MAX_INPUT_BYTES = 4 * 1024 * 1024 * 1024
ATTEMPT_SENTINEL_PATH = "/work/cache/ferrl-attack-sentinel"
PARENT_DEVICE_CANARY = b"ferrl-parent-private-cuda-v1-7f4c3a19"
PAYLOAD_WIRE_PREFIX = b"FERRL-PAYLOAD-v1\0"
HARDENING_WIRE_PREFIX = b"HARDENED-v1\0"
HARDENING_CONTRACT = "ferrl.candidate-hardening.v1"
DEVICE_IDENTITY_CONTRACT = "ferrl.executing-device.v1"
SECCOMP_POLICY = "x86_64-tsync-af-unix-v1"


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.prctl.restype = ctypes.c_int
_LIBC.syscall.restype = ctypes.c_long


class _CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def _prctl(option, value):
    if _LIBC.prctl(option, value, 0, 0, 0) != 0:
        raise SystemExit(114)


def _prctl_value(option):
    value = _LIBC.prctl(option, 0, 0, 0, 0)
    if value < 0:
        raise RuntimeError(f"prctl query {option} failed with errno {ctypes.get_errno()}")
    return value


def _cuda_driver_call(function, *args):
    result = function(*args)
    if result != 0:
        raise InfrastructureFailure(f"CUDA driver identity query failed with code {result}")


def _executing_device_identity():
    """Return canonical identity for the CUDA device used by the trusted controller."""
    try:
        logical_ordinal = int(torch.cuda.current_device())
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDeviceGetName.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
            ctypes.c_int,
        ]
        driver.cuDeviceGetName.restype = ctypes.c_int
        driver.cuDeviceGetPCIBusId.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
            ctypes.c_int,
        ]
        driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
        uuid_query = getattr(driver, "cuDeviceGetUuid_v2", None)
        if uuid_query is None:
            uuid_query = driver.cuDeviceGetUuid
        uuid_query.argtypes = [ctypes.POINTER(_CUuuid), ctypes.c_int]
        uuid_query.restype = ctypes.c_int

        _cuda_driver_call(driver.cuInit, 0)
        device = ctypes.c_int()
        _cuda_driver_call(driver.cuDeviceGet, ctypes.byref(device), logical_ordinal)
        name_buffer = ctypes.create_string_buffer(256)
        pci_buffer = ctypes.create_string_buffer(64)
        uuid_value = _CUuuid()
        _cuda_driver_call(driver.cuDeviceGetName, name_buffer, len(name_buffer), device)
        _cuda_driver_call(
            driver.cuDeviceGetPCIBusId,
            pci_buffer,
            len(pci_buffer),
            device,
        )
        _cuda_driver_call(uuid_query, ctypes.byref(uuid_value), device)
        name = name_buffer.value.decode("utf-8", errors="strict").strip()
        pci_bus_id = pci_buffer.value.decode("ascii", errors="strict").strip().lower()
        uuid = bytes(uuid_value.bytes).hex()
        if not name or not pci_bus_id or len(uuid) != 32:
            raise InfrastructureFailure("CUDA driver returned incomplete device identity")
        return json.dumps(
            {
                "contract": DEVICE_IDENTITY_CONTRACT,
                "cuda_logical_ordinal": logical_ordinal,
                "name": name,
                "pci_bus_id": pci_bus_id,
                "uuid": uuid,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    except InfrastructureFailure:
        raise
    except BaseException as error:
        raise InfrastructureFailure("trusted CUDA device identity query failed") from error


# Arm procfs protection before any verifier imports. Spawned candidate workers
# repeat this at their target entry; they never inherit the grade socket.
_prctl(PR_SET_DUMPABLE, 0)
sys.path.insert(0, "/opt/ferrl-verifier")

import eval as upstream  # noqa: E402
import torch  # noqa: E402
import torch.cuda  # noqa: E402
from reference import check_implementation as _CHECK_IMPLEMENTATION  # noqa: E402
from reference import generate_input as _GENERATE_INPUT  # noqa: E402


_CLONE_DATA = upstream._clone_data
_CALCULATE_STATS = upstream.calculate_stats
_GET_TEST_CASES = upstream.get_test_cases
_SET_SEED = upstream.set_seed
_STATS_TYPE = upstream.Stats
_SYNC = torch.cuda.synchronize
_TENSOR_TYPE = torch.Tensor
_COPY_TENSOR = torch.Tensor.copy_


class InfrastructureFailure(Exception):
    pass


class CandidateFailure(Exception):
    pass


class GradeLogger:
    def __init__(self, path):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.set_inheritable(False)
        self.socket.connect(path)
        self.file = self.socket.makefile("w", encoding="utf-8", newline="\n")

    def close(self):
        try:
            self.file.close()
        finally:
            self.socket.close()

    def raw(self, value):
        print(value, file=self.file, flush=True)

    def log(self, key, value):
        self.raw(f"{key}: {value}")


def _bounded_message(message):
    text = str(message).replace("\r", " ").replace("\n", " ")
    return text[:2048]


def _consume_attempt_sentinel(logger, phase, index):
    text = None
    try:
        descriptor = os.open(
            ATTEMPT_SENTINEL_PATH,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        try:
            metadata = os.fstat(descriptor)
            if stat.S_ISREG(metadata.st_mode) and 0 < metadata.st_size <= 256:
                value = os.read(descriptor, 257)
                if len(value) == metadata.st_size:
                    candidate_text = value.decode("ascii")
                    if not any(character in candidate_text for character in "\r\n\0"):
                        text = candidate_text
        finally:
            os.close(descriptor)
    except (OSError, UnicodeError):
        pass
    try:
        os.unlink(ATTEMPT_SENTINEL_PATH)
    except OSError:
        pass
    if text is not None:
        logger.log(f"{phase}.{index}.candidate-sentinel", text)


def _send_status(connection, value):
    try:
        connection.send_bytes(value)
    except BaseException:
        os._exit(115)


def _send_payload(connection, event, payload=b""):
    try:
        connection.send_bytes(PAYLOAD_WIRE_PREFIX + event + b"\0" + payload)
    except BaseException:
        os._exit(115)


def _recv_payload(connection, limit):
    value = connection.recv_bytes(limit)
    if not value.startswith(PAYLOAD_WIRE_PREFIX):
        raise ValueError("candidate payload wire prefix mismatch")
    event, separator, payload = value[len(PAYLOAD_WIRE_PREFIX) :].partition(b"\0")
    if not separator or not event:
        raise ValueError("candidate payload wire frame is malformed")
    return event, payload


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ushort),
        ("filters", ctypes.POINTER(_SockFilter)),
    ]


def _candidate_seccomp_filter():
    # Classic BPF over `struct seccomp_data`. This policy is intentionally a
    # narrow deny list: CUDA, PyTorch, Triton, ptxas, and their compiler children
    # retain ordinary compute/file/process syscalls, while cross-process access,
    # namespace changes, kernel attack surfaces, and non-AF_UNIX sockets are
    # removed. The architecture check prevents syscall-number confusion.
    bpf_ld_w_abs = 0x20
    bpf_jmp_jeq_k = 0x15
    bpf_jmp_jset_k = 0x45
    bpf_ret_k = 0x06
    seccomp_ret_kill_process = 0x80000000
    seccomp_ret_errno = 0x00050000
    seccomp_ret_allow = 0x7FFF0000
    audit_arch_x86_64 = 0xC000003E
    x32_syscall_bit = 0x40000000
    errno_eperm = 1
    errno_enosys = 38

    def stmt(code, value):
        return (code, 0, 0, value)

    def jump(code, value, yes, no):
        return (code, yes, no, value)

    deny = [
        101,  # ptrace
        155,  # pivot_root
        161,  # chroot
        165,  # mount
        166,  # umount2
        248,  # add_key
        249,  # request_key
        250,  # keyctl
        272,  # unshare
        298,  # perf_event_open
        303,  # name_to_handle_at
        304,  # open_by_handle_at
        308,  # setns
        310,  # process_vm_readv
        311,  # process_vm_writev
        312,  # kcmp
        321,  # bpf
        323,  # userfaultfd
        424,  # pidfd_send_signal
        425,  # io_uring_setup
        426,  # io_uring_enter
        427,  # io_uring_register
        428,  # open_tree
        429,  # move_mount
        430,  # fsopen
        431,  # fsconfig
        432,  # fsmount
        433,  # fspick
        434,  # pidfd_open
        438,  # pidfd_getfd
        440,  # process_madvise
        442,  # mount_setattr
        448,  # process_mrelease
    ]
    instructions = [
        stmt(bpf_ld_w_abs, 4),
        jump(bpf_jmp_jeq_k, audit_arch_x86_64, 1, 0),
        stmt(bpf_ret_k, seccomp_ret_kill_process),
        stmt(bpf_ld_w_abs, 0),
        # Reject x32 syscalls, which share AUDIT_ARCH_X86_64 but use a distinct
        # syscall-number space not covered by the x86_64 table below.
        jump(bpf_jmp_jset_k, x32_syscall_bit, 0, 1),
        stmt(bpf_ret_k, seccomp_ret_errno | errno_eperm),
    ]
    for syscall_number in deny:
        instructions.extend(
            [
                jump(bpf_jmp_jeq_k, syscall_number, 0, 1),
                stmt(bpf_ret_k, seccomp_ret_errno | errno_eperm),
            ]
        )

    # clone3 hides its flags behind a user pointer, which classic seccomp BPF
    # cannot inspect. Report ENOSYS so libc/subprocess falls back to clone/vfork;
    # clone's direct flag argument is filtered below.
    instructions.extend(
        [
            jump(bpf_jmp_jeq_k, 435, 0, 1),
            stmt(bpf_ret_k, seccomp_ret_errno | errno_enosys),
        ]
    )

    clone_new_namespace_mask = 0x7E020000
    instructions.extend(
        [
            # If this is not clone(2), skip the argument check and reload the
            # syscall number for the socket checks.
            jump(bpf_jmp_jeq_k, 56, 0, 3),
            stmt(bpf_ld_w_abs, 16),
            jump(bpf_jmp_jset_k, clone_new_namespace_mask, 0, 1),
            stmt(bpf_ret_k, seccomp_ret_errno | errno_eperm),
            stmt(bpf_ld_w_abs, 0),
            # socket(2) and socketpair(2) are permitted only for AF_UNIX. The
            # verifier protocol and multiprocessing pipes need local sockets;
            # no IP, packet, or netlink socket can be created after this point.
            jump(bpf_jmp_jeq_k, 41, 1, 0),
            jump(bpf_jmp_jeq_k, 53, 0, 3),
            stmt(bpf_ld_w_abs, 16),
            jump(bpf_jmp_jeq_k, socket.AF_UNIX, 1, 0),
            stmt(bpf_ret_k, seccomp_ret_errno | errno_eperm),
            stmt(bpf_ret_k, seccomp_ret_allow),
        ]
    )
    return instructions


def _read_process_status():
    wanted = {
        "CapInh",
        "CapPrm",
        "CapEff",
        "CapBnd",
        "CapAmb",
        "NoNewPrivs",
        "Seccomp",
        "Seccomp_filters",
    }
    values = {}
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            key, separator, value = line.partition(":")
            if separator and key in wanted:
                values[key] = value.strip()
    return values


def _validated_hardening_record(payload):
    if len(payload) > MAX_STATUS_BYTES - len(HARDENING_WIRE_PREFIX):
        raise ValueError("candidate hardening evidence exceeds its wire cap")
    try:
        text = payload.decode("ascii")
        evidence = json.loads(text)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("candidate hardening evidence is malformed") from error
    expected_keys = {
        "arch",
        "cap_amb",
        "cap_bnd",
        "cap_eff",
        "cap_inh",
        "cap_prm",
        "cgroup",
        "contract",
        "dumpable",
        "denial_probes",
        "landlock",
        "network_socket_policy",
        "no_new_privs",
        "physical_gpu_isolation",
        "seccomp_filters",
        "seccomp_mode",
        "seccomp_policy",
        "seccomp_tsync",
        "unix_socket_probe",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        raise ValueError("candidate hardening evidence schema mismatch")
    if any(
        type(evidence[key]) is not int
        for key in ("dumpable", "no_new_privs", "seccomp_mode")
    ):
        raise ValueError("candidate hardening evidence has a non-integer kernel field")
    if (
        evidence["contract"] != HARDENING_CONTRACT
        or evidence["arch"] != "x86_64"
        or evidence["dumpable"] != 0
        or evidence["no_new_privs"] != 1
        or evidence["seccomp_mode"] != 2
        or evidence["seccomp_policy"] != SECCOMP_POLICY
        or evidence["seccomp_tsync"] is not True
        or evidence["denial_probes"]
        != [
            "bpf",
            "io_uring",
            "namespace",
            "network",
            "parent_proc",
            "pidfd_getfd",
            "process_vm",
            "ptrace",
        ]
        or evidence["unix_socket_probe"] is not True
        or evidence["network_socket_policy"] != "af_unix_only"
        or evidence["landlock"] is not False
        or evidence["cgroup"] is not False
        or evidence["physical_gpu_isolation"] is not False
    ):
        raise ValueError("candidate hardening evidence did not prove the required controls")
    if evidence["seccomp_filters"] is not None and (
        type(evidence["seccomp_filters"]) is not int or evidence["seccomp_filters"] < 1
    ):
        raise ValueError("candidate hardening evidence has an invalid filter count")
    for key in ("cap_amb", "cap_bnd", "cap_eff", "cap_inh", "cap_prm"):
        value = evidence[key]
        if (
            not isinstance(value, str)
            or len(value) != 16
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("candidate hardening capability evidence is malformed")
    if any(int(evidence[key], 16) != 0 for key in ("cap_amb", "cap_eff", "cap_inh", "cap_prm")):
        raise ValueError("candidate retained an active or inheritable capability")
    forbidden_bounding_caps = (1 << 19) | (1 << 21)  # CAP_SYS_PTRACE | CAP_SYS_ADMIN
    if int(evidence["cap_bnd"], 16) & forbidden_bounding_caps:
        raise ValueError("candidate bounding set retained ptrace or admin capability")
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    if canonical != text:
        raise ValueError("candidate hardening evidence is not canonical JSON")
    return text


def _install_candidate_hardening():
    if os.uname().machine != "x86_64":
        raise RuntimeError("candidate seccomp policy requires x86_64")
    _prctl(PR_SET_NO_NEW_PRIVS, 1)
    if _prctl_value(PR_GET_NO_NEW_PRIVS) != 1:
        raise RuntimeError("candidate no_new_privs did not become irreversible")

    policy = _candidate_seccomp_filter()
    filters = (_SockFilter * len(policy))(*(_SockFilter(*instruction) for instruction in policy))
    program = _SockFprog(
        length=len(filters),
        filters=ctypes.cast(filters, ctypes.POINTER(_SockFilter)),
    )
    sys_seccomp = 317
    seccomp_set_mode_filter = 1
    seccomp_filter_flag_tsync = 1
    ctypes.set_errno(0)
    result = _LIBC.syscall(
        ctypes.c_long(sys_seccomp),
        ctypes.c_ulong(seccomp_set_mode_filter),
        ctypes.c_ulong(seccomp_filter_flag_tsync),
        ctypes.byref(program),
    )
    if result != 0:
        raise RuntimeError(f"candidate seccomp TSYNC failed with errno {ctypes.get_errno()}")

    def require_eperm(syscall_number, *arguments):
        ctypes.set_errno(0)
        value = _LIBC.syscall(
            ctypes.c_long(syscall_number),
            *(ctypes.c_long(argument) for argument in arguments),
        )
        if value >= 0:
            # The socket probe could otherwise leak a live network descriptor
            # before this trusted initialization aborts.
            if syscall_number == 41:
                os.close(value)
            raise RuntimeError(f"candidate syscall {syscall_number} escaped seccomp")
        if ctypes.get_errno() != 1:
            raise RuntimeError(
                f"candidate syscall {syscall_number} returned unexpected errno {ctypes.get_errno()}"
            )

    # Non-vacuous probes execute representative rules after TSYNC. Arguments are
    # chosen to have no side effect if a rule were accidentally omitted.
    require_eperm(101, -1, 0, 0, 0)  # invalid ptrace request
    require_eperm(310, os.getpid(), 0, 0, 0, 0, 0)  # process_vm_readv
    require_eperm(438, -1, -1, 0)  # pidfd_getfd
    require_eperm(272, 0)  # unshare with an empty flag set
    require_eperm(321, -1, 0, 0)  # invalid bpf command
    require_eperm(425, 0, 0)  # io_uring_setup
    require_eperm(41, socket.AF_INET, socket.SOCK_STREAM, 0)
    unix_left, unix_right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    unix_left.close()
    unix_right.close()

    parent_mem = f"/proc/{os.getppid()}/mem"
    try:
        descriptor = os.open(
            parent_mem,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM):
            raise RuntimeError(
                f"candidate parent-memory probe returned unexpected errno {error.errno}"
            ) from error
    else:
        os.close(descriptor)
        raise RuntimeError("candidate could open controller process memory")

    status = _read_process_status()
    required = {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs", "Seccomp"}
    if not required.issubset(status):
        raise RuntimeError("candidate privilege evidence is incomplete")
    if int(status["NoNewPrivs"]) != 1 or int(status["Seccomp"]) != 2:
        raise RuntimeError("candidate kernel hardening is not active")
    for key in ("CapInh", "CapPrm", "CapEff", "CapAmb"):
        if int(status[key], 16) != 0:
            raise RuntimeError(f"candidate retained capability set {key}")
    forbidden_bounding_caps = (1 << 19) | (1 << 21)  # CAP_SYS_PTRACE | CAP_SYS_ADMIN
    if int(status["CapBnd"], 16) & forbidden_bounding_caps:
        raise RuntimeError("candidate bounding set retained ptrace or admin capability")
    seccomp_filters = status.get("Seccomp_filters")
    if seccomp_filters is not None:
        seccomp_filters = int(seccomp_filters)
        if seccomp_filters < 1:
            raise RuntimeError("candidate seccomp filter count is invalid")
    evidence = {
        "arch": "x86_64",
        "cap_amb": status["CapAmb"].lower().zfill(16),
        "cap_bnd": status["CapBnd"].lower().zfill(16),
        "cap_eff": status["CapEff"].lower().zfill(16),
        "cap_inh": status["CapInh"].lower().zfill(16),
        "cap_prm": status["CapPrm"].lower().zfill(16),
        "cgroup": False,
        "contract": HARDENING_CONTRACT,
        "denial_probes": [
            "bpf",
            "io_uring",
            "namespace",
            "network",
            "parent_proc",
            "pidfd_getfd",
            "process_vm",
            "ptrace",
        ],
        "dumpable": _prctl_value(PR_GET_DUMPABLE),
        "landlock": False,
        "network_socket_policy": "af_unix_only",
        "no_new_privs": int(status["NoNewPrivs"]),
        "physical_gpu_isolation": False,
        "seccomp_filters": seccomp_filters,
        "seccomp_mode": int(status["Seccomp"]),
        "seccomp_policy": SECCOMP_POLICY,
        "seccomp_tsync": True,
        "unix_socket_probe": True,
    }
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("ascii")
    _validated_hardening_record(payload)
    return payload


def _cpu_clone(value):
    if type(value) is _TENSOR_TYPE:
        return value.detach().to(device="cpu").clone()
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise InfrastructureFailure("trusted input contained an unsupported value")


def _cuda_clone(value):
    if type(value) is _TENSOR_TYPE:
        return value.to(device="cuda")
    if isinstance(value, tuple):
        return tuple(_cuda_clone(item) for item in value)
    if isinstance(value, list):
        return [_cuda_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _cuda_clone(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("candidate input contained an unsupported value")


def _serialize_input(value):
    buffer = io.BytesIO()
    torch.save(_cpu_clone(value), buffer)
    payload = buffer.getvalue()
    if len(payload) > MAX_INPUT_BYTES:
        raise InfrastructureFailure("trusted candidate input exceeds transport cap")
    return payload


def _deserialize_input(payload):
    # The protected parent created these bytes. weights_only still narrows the
    # accepted graph before it enters the disposable candidate interpreter.
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    return _cuda_clone(value)


def _tensor_bytes(value):
    cpu = value.detach().to(device="cpu").contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def _candidate_payload(commands, results):
    _prctl(PR_SET_DUMPABLE, 0)
    try:
        os.setsid()
    except OSError:
        pass

    # Capture exact trusted callables before candidate import can replace module
    # globals in this disposable process.
    sync = _SYNC
    tensor_type = _TENSOR_TYPE
    try:
        torch.cuda.init()
        sync()
    except BaseException:
        _send_payload(results, b"TRUSTED_INIT_ERROR")
        return
    try:
        hardening = _install_candidate_hardening()
    except BaseException:
        _send_payload(results, b"HARDENING_ERROR")
        return
    _send_payload(results, b"HARDENED", hardening)
    _send_payload(results, b"READY")

    try:
        if commands.recv_bytes(MAX_STATUS_BYTES) != b"ACK-v2":
            return
    except BaseException:
        return

    entry_sent = False

    def trace_candidate_entry(frame, event, arg):
        nonlocal entry_sent
        del arg
        if (
            not entry_sent
            and event == "call"
            and frame.f_code.co_filename == SUBMISSION_PATH
        ):
            entry_sent = True
            _send_payload(results, b"ENTRY")
            try:
                if commands.recv_bytes(MAX_STATUS_BYTES) != ENTRY_ACK:
                    raise RuntimeError("candidate entry acknowledgement failed")
            except BaseException as error:
                raise RuntimeError("candidate entry was not acknowledged") from error
            sys.settrace(None)
            return None
        return trace_candidate_entry

    try:
        sys.modules.pop("submission", None)
        sys.settrace(trace_candidate_entry)
        module = importlib.import_module("submission")
        sys.settrace(None)
        kernel = module.custom_kernel
        if not callable(kernel):
            raise TypeError("custom_kernel is not callable")
    except BaseException:
        sys.settrace(None)
        _send_payload(results, b"IMPORT_ERROR" if entry_sent else b"IMPORT_REJECTED")
        return
    if not entry_sent:
        _send_payload(results, b"IMPORT_REJECTED")
        return
    _send_payload(results, b"IMPORTED")

    while True:
        try:
            command = commands.recv_bytes(MAX_INPUT_BYTES + 16)
        except EOFError:
            return
        except BaseException:
            _send_payload(results, b"CONTROL_ERROR")
            return
        if command == b"CLOSE-v2":
            return
        prefix = b"RUN-v2\0"
        if not command.startswith(prefix):
            _send_payload(results, b"CONTROL_ERROR")
            return
        try:
            candidate_data = _deserialize_input(command[len(prefix) :])
            if not isinstance(candidate_data, (tuple, list)) or not candidate_data:
                raise TypeError("candidate input schema mismatch")
            expected = candidate_data[0]
            if type(expected) is not tensor_type or not expected.is_cuda:
                raise TypeError("candidate output schema source is not a CUDA tensor")
            output = kernel(candidate_data)
            if type(output) is not tensor_type:
                raise TypeError("candidate output is not an exact torch.Tensor")
            if (
                output.shape != expected.shape
                or output.dtype != expected.dtype
                or output.device != expected.device
            ):
                raise ValueError("candidate output schema mismatch")
            sync()
            raw_output = _tensor_bytes(output)
        except BaseException:
            _send_payload(results, b"CANDIDATE_ERROR")
            continue
        _send_payload(results, b"OUTPUT", raw_output)


def _read_parent_pid(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("PPid:\t"):
                    return int(line.split(":", 1)[1])
    except (OSError, ValueError):
        return None
    return None


def _is_descendant(pid, root):
    for _ in range(256):
        if pid == root:
            return True
        parent = _read_parent_pid(pid)
        if parent is None or parent <= 1 or parent == pid:
            return False
        pid = parent
    return False


def _kill_candidate_tree():
    self_pid = os.getpid()
    for _ in range(8):
        victims = []
        try:
            entries = os.listdir("/proc")
        except OSError:
            return
        for entry in entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid not in (1, self_pid) and _is_descendant(pid, self_pid):
                victims.append(pid)
        if not victims:
            return
        for pid in sorted(victims, reverse=True):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        time.sleep(0.01)


def _candidate_worker(commands, status, outputs):
    _prctl(PR_SET_DUMPABLE, 0)
    try:
        os.setsid()
    except OSError:
        pass

    context = multiprocessing.get_context("spawn")
    child_commands, payload_commands = context.Pipe(duplex=False)
    payload_results, child_results = context.Pipe(duplex=False)
    payload_process = context.Process(
        target=_candidate_payload,
        args=(child_commands, child_results),
        daemon=False,
    )
    payload_process.start()
    child_commands.close()
    child_results.close()
    try:
        phase = "preparation"
        try:
            event, payload = _recv_payload(payload_results, MAX_STATUS_BYTES)
        except BaseException:
            _send_status(status, b"TRUSTED_INIT_ERROR")
            return
        if event != b"HARDENED":
            _send_status(status, b"TRUSTED_INIT_ERROR")
            return
        try:
            hardening = _validated_hardening_record(payload).encode("ascii")
        except BaseException:
            _send_status(status, b"TRUSTED_INIT_ERROR")
            return
        _send_status(status, HARDENING_WIRE_PREFIX + hardening)

        try:
            event, payload = _recv_payload(payload_results, MAX_STATUS_BYTES)
        except BaseException:
            _send_status(status, b"TRUSTED_INIT_ERROR")
            return
        if event != b"READY" or payload:
            _send_status(status, b"TRUSTED_INIT_ERROR")
            return
        _send_status(status, b"READY")

        try:
            acknowledgement = commands.recv_bytes(MAX_STATUS_BYTES)
        except BaseException:
            return
        if acknowledgement != b"ACK-v2":
            return
        payload_commands.send_bytes(acknowledgement)

        entered = False
        import_events = 0

        def reject_import_protocol():
            _send_status(status, b"IMPORT_ERROR" if entered else b"IMPORT_REJECTED")

        while True:
            import_events += 1
            if import_events > MAX_STATUS_EVENTS:
                reject_import_protocol()
                return
            try:
                event, payload = _recv_payload(payload_results, MAX_STATUS_BYTES)
            except BaseException:
                reject_import_protocol()
                return
            if payload:
                reject_import_protocol()
                return
            if event == b"ENTRY":
                if entered:
                    reject_import_protocol()
                    return
                entered = True
                _send_status(status, b"ENTRY")
                try:
                    acknowledgement = commands.recv_bytes(MAX_STATUS_BYTES)
                    if acknowledgement != ENTRY_ACK:
                        reject_import_protocol()
                        return
                    payload_commands.send_bytes(acknowledgement)
                except BaseException:
                    return
                continue
            if event == b"IMPORTED":
                _send_status(status, event)
                break
            # The sealed payload wrapper emits IMPORT_REJECTED only before its
            # traced submission entry. Seeing it after acknowledged entry proves
            # candidate code wrote the real payload-results channel.
            if event == b"IMPORT_REJECTED" and entered:
                _send_status(status, b"IMPORT_CHANNEL_CORRUPTED")
                return
            if event in (b"IMPORT_REJECTED", b"IMPORT_ERROR"):
                reject_import_protocol()
                return
            reject_import_protocol()
            return

        while True:
            try:
                command = commands.recv_bytes(MAX_INPUT_BYTES + 16)
            except BaseException:
                return
            if command == b"CLOSE-v2":
                try:
                    payload_commands.send_bytes(command)
                except BaseException:
                    pass
                return
            if not command.startswith(b"RUN-v2\0"):
                _send_status(status, b"CONTROL_ERROR")
                return
            try:
                payload_commands.send_bytes(command)
                event, payload = _recv_payload(
                    payload_results,
                    MAX_INPUT_BYTES + len(PAYLOAD_WIRE_PREFIX) + 32,
                )
            except BaseException:
                _send_status(status, b"CANDIDATE_ERROR")
                continue
            if event == b"CANDIDATE_ERROR" and not payload:
                _send_status(status, b"CANDIDATE_ERROR")
                continue
            if event != b"OUTPUT":
                _send_status(status, b"CONTROL_ERROR")
                return
            _send_status(status, b"OUTPUT_READY")
            try:
                outputs.send_bytes(payload)
            except BaseException:
                return
    finally:
        payload_commands.close()
        payload_results.close()
        payload_process.join(0.25)
        if payload_process.is_alive():
            payload_process.kill()
            payload_process.join(1)


class CandidateSession:
    def __init__(self, mode, logger, isolation_tier, timing_metric):
        self.mode = mode
        self.logger = logger
        self.isolation_tier = isolation_tier
        self.timing_metric = timing_metric
        self.entered = False
        self.rejected = False
        self.closed = False
        self.import_status_events = 0
        context = multiprocessing.get_context("spawn")
        child_commands, self.commands = context.Pipe(duplex=False)
        self.status, child_status = context.Pipe(duplex=False)
        self.outputs, child_outputs = context.Pipe(duplex=False)
        self.worker = context.Process(
            target=_candidate_worker,
            args=(child_commands, child_status, child_outputs),
            daemon=False,
        )
        self.worker.start()
        child_commands.close()
        child_status.close()
        child_outputs.close()
        try:
            hardening = self._recv_import(before_entry=True)
            if not hardening.startswith(HARDENING_WIRE_PREFIX):
                raise InfrastructureFailure("candidate worker hardening proof was missing")
            hardening = hardening[len(HARDENING_WIRE_PREFIX) :]
            hardening = _validated_hardening_record(hardening)
            self.logger.log("ferrl-candidate-hardening", hardening)
            if self._recv_import(before_entry=True) != b"READY":
                raise InfrastructureFailure("candidate worker trusted initialization failed")
            self.logger.log("ferrl-verifier-isolation-tier", self.isolation_tier)
            self.logger.log("ferrl-timing-metric", self.timing_metric)
            self.commands.send_bytes(b"ACK-v2")
            while True:
                event = self._recv_import(before_entry=not self.entered)
                if event == b"ENTRY" and not self.entered:
                    self.entered = True
                    self.logger.log("ferrl-entry", f"{self.mode}-v4")
                    self.commands.send_bytes(ENTRY_ACK)
                    continue
                if event == b"ENTRY":
                    raise CandidateFailure("candidate worker sent duplicate entry")
                if event == b"IMPORTED" and self.entered:
                    break
                if event == b"IMPORT_REJECTED" and not self.entered:
                    self.rejected = True
                    self.logger.log("ferrl-candidate-rejected", f"{self.mode}-import-v1")
                    break
                if event == b"IMPORT_CHANNEL_CORRUPTED" and self.entered:
                    self.rejected = True
                    self.logger.log("ferrl-candidate-rejected", f"{self.mode}-import-v1")
                    self.logger.log(
                        "ferrl-candidate-rejection-reason",
                        "payload-results-channel-v1",
                    )
                    break
                if event == b"IMPORT_ERROR" and self.entered:
                    self.rejected = True
                    self.logger.log("ferrl-candidate-rejected", f"{self.mode}-import-v1")
                    self.logger.log("ferrl-candidate-rejection-reason", "protocol-v1")
                    break
                raise InfrastructureFailure("candidate worker import protocol failed")
        except CandidateFailure as error:
            if not self.entered:
                self.close()
                raise
            self.rejected = True
            self.logger.log("ferrl-candidate-rejected", f"{self.mode}-import-v1")
            reason = (
                "controller-loss-v1"
                if str(error) == "candidate worker exited after acknowledged entry"
                else "protocol-v1"
            )
            self.logger.log("ferrl-candidate-rejection-reason", reason)
            self.close()
        except BaseException:
            self.close()
            raise

    def _recv_import(self, before_entry=False):
        self.import_status_events += 1
        if self.import_status_events > MAX_STATUS_EVENTS:
            raise CandidateFailure("candidate worker status flood")
        return self._recv(before_entry=before_entry)

    def _recv(self, before_entry=False):
        try:
            return self.status.recv_bytes(MAX_STATUS_BYTES)
        except (EOFError, OSError) as error:
            if before_entry:
                raise InfrastructureFailure("candidate worker exited before entry") from error
            raise CandidateFailure("candidate worker exited after acknowledged entry") from error

    def execute(self, candidate_data, checked_output):
        if self.rejected:
            raise CandidateFailure("candidate import was rejected")
        try:
            payload = _serialize_input(candidate_data)
        except InfrastructureFailure:
            raise
        except BaseException as error:
            raise InfrastructureFailure("trusted input serialization failed") from error

        # No CUDA object crosses this boundary. A candidate may corrupt its own
        # pipes or emit OUTPUT_READY early, but it cannot earn correctness without
        # supplying an exact-size CPU result that passes the protected checker.
        started = time.perf_counter_ns()
        try:
            self.commands.send_bytes(b"RUN-v2\0" + payload)
        except BaseException as error:
            raise CandidateFailure("candidate input handoff failed") from error
        status = self._recv()
        if status != b"OUTPUT_READY":
            raise CandidateFailure("candidate execution failed")
        expected_bytes = checked_output.numel() * checked_output.element_size()
        try:
            raw_output = self.outputs.recv_bytes(expected_bytes)
        except (EOFError, OSError) as error:
            raise CandidateFailure("candidate output transport failed") from error
        elapsed = time.perf_counter_ns() - started
        if len(raw_output) != expected_bytes:
            raise CandidateFailure("candidate output byte length mismatch")
        try:
            cpu_output = torch.frombuffer(
                bytearray(raw_output),
                dtype=checked_output.dtype,
                count=checked_output.numel(),
            ).reshape(checked_output.shape)
            _COPY_TENSOR(checked_output, cpu_output)
            _SYNC()
        except BaseException as error:
            raise InfrastructureFailure("trusted output reconstruction failed") from error
        return elapsed

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.worker.is_alive():
                self.commands.send_bytes(b"CLOSE-v2")
        except BaseException:
            pass
        self.worker.join(0.25)
        if self.worker.is_alive():
            self.worker.kill()
            self.worker.join(1)
        self.commands.close()
        self.status.close()
        self.outputs.close()
        _kill_candidate_tree()


def _wrap_check(data, output):
    result = _CHECK_IMPLEMENTATION(data, output)
    if isinstance(result, tuple):
        return result
    return not bool(result), result


def _execute_checked(session, args):
    try:
        parent_device_canary = torch.tensor(
            list(PARENT_DEVICE_CANARY), dtype=torch.uint8, device="cuda"
        )
        data = _GENERATE_INPUT(**args)
        candidate_data = _CLONE_DATA(data)
        checked_output = torch.full_like(data[0], float("nan"))
    except BaseException as error:
        raise InfrastructureFailure("trusted input/output preparation failed") from error
    elapsed = session.execute(candidate_data, checked_output)
    try:
        good, message = _wrap_check(data, checked_output)
    except BaseException as error:
        raise InfrastructureFailure("trusted correctness checker failed") from error
    if _tensor_bytes(parent_device_canary) != PARENT_DEVICE_CANARY:
        raise InfrastructureFailure("parent-private CUDA canary changed")
    return bool(good), _bounded_message(message or ""), elapsed


def _run_testing(logger, tests, isolation_tier, timing_metric, device_identity):
    if not tests:
        raise InfrastructureFailure("trusted test case set is empty")
    session = CandidateSession("test", logger, isolation_tier, timing_metric)
    try:
        logger.log("ferrl-executing-device", device_identity)
        logger.log("test-count", len(tests))
        if session.rejected:
            _consume_attempt_sentinel(logger, "test", "import")
            logger.log("check", "fail")
            logger.log("test-exit", 112)
            return False
        passed = True
        for index, test in enumerate(tests):
            logger.log(f"test.{index}.spec", test.spec)
            try:
                good, message, _ = _execute_checked(session, dict(test.args))
            except CandidateFailure:
                good, message = False, "candidate execution failed"
            _consume_attempt_sentinel(logger, "test", index)
            logger.log(f"test.{index}.status", "pass" if good else "fail")
            if message:
                logger.log(
                    f"test.{index}.message" if good else f"test.{index}.error",
                    message,
                )
            if not good:
                passed = False
        logger.log("check", "pass" if passed else "fail")
        logger.log("test-exit", 0 if passed else 112)
        return passed
    finally:
        session.close()


def _run_benchmark_case(session, test):
    # One checked, untimed call warms compilation and the exact shape.
    good, message, _ = _execute_checked(session, dict(test.args))
    if not good:
        return message or "candidate benchmark warmup was incorrect"

    durations = []
    benchmark_started = time.perf_counter_ns()
    for iteration in range(100):
        args = dict(test.args)
        if "seed" in args:
            args["seed"] += 13 * (iteration + 1)
        good, message, elapsed = _execute_checked(session, args)
        if not good:
            return message or "candidate benchmark output was incorrect"
        durations.append(elapsed)
        if iteration > 1:
            stats = _CALCULATE_STATS(durations)
            wall = time.perf_counter_ns() - benchmark_started
            if (
                stats.err / stats.mean < 0.001
                or stats.mean * stats.runs > 10e9
                or wall > 120e9
            ):
                break
    return _CALCULATE_STATS(durations)


def _run_benchmarking(logger, tests, isolation_tier, timing_metric, device_identity):
    if not tests:
        raise InfrastructureFailure("trusted benchmark case set is empty")
    session = CandidateSession("benchmark", logger, isolation_tier, timing_metric)
    try:
        logger.log("ferrl-executing-device", device_identity)
        logger.log("benchmark-count", len(tests))
        if session.rejected:
            logger.log("check", "fail")
            logger.log("benchmark-exit", 112)
            return False
        passed = True
        for index, test in enumerate(tests):
            logger.log(f"benchmark.{index}.spec", test.spec)
            try:
                result = _run_benchmark_case(session, test)
            except CandidateFailure:
                result = "candidate benchmark execution failed"
            _consume_attempt_sentinel(logger, "benchmark", index)
            if isinstance(result, _STATS_TYPE):
                for field in ("runs", "mean", "std", "err", "best", "worst"):
                    logger.log(f"benchmark.{index}.{field}", getattr(result, field))
            else:
                passed = False
                logger.log(f"benchmark.{index}.status", "fail")
                logger.log(f"benchmark.{index}.error", _bounded_message(result))
        logger.log("check", "pass" if passed else "fail")
        logger.log("benchmark-exit", 0 if passed else 112)
        return passed
    finally:
        session.close()


def main():
    if len(sys.argv) != 3:
        return 114
    grade_socket = os.environ.pop("FERRL_GRADE_SOCKET", None)
    if not grade_socket:
        return 114
    isolation_tier = os.environ.pop("FERRL_VERIFIER_ISOLATION_TIER", None)
    timing_metric = os.environ.pop("FERRL_TIMING_METRIC", None)
    seed_raw = os.environ.pop("POPCORN_SEED", None)

    try:
        logger = GradeLogger(grade_socket)
    except BaseException:
        return 114

    try:
        phase = "preparation"
        try:
            try:
                seed = int(seed_raw) if seed_raw is not None else None
            except ValueError as error:
                raise InfrastructureFailure(
                    "trusted case-generation seed is not an integer"
                ) from error
            if seed is not None and not 0 <= seed <= MAX_CASE_SEED:
                raise InfrastructureFailure(
                    "trusted case-generation seed is outside unsigned 32-bit range"
                )
            if (
                isolation_tier not in ISOLATION_TIMING_METRICS
                or ISOLATION_TIMING_METRICS[isolation_tier] != timing_metric
            ):
                raise InfrastructureFailure("trusted verifier isolation/timing contract mismatch")
            _prctl(PR_SET_CHILD_SUBREAPER, 1)
            _SET_SEED(42 if seed is None else seed)
            torch.cuda.init()
            _SYNC()
            device_identity = _executing_device_identity()
            test_cases = _GET_TEST_CASES(sys.argv[1], seed)
            benchmark_cases = _GET_TEST_CASES(sys.argv[2], seed)
            if not test_cases or not benchmark_cases:
                raise InfrastructureFailure("trusted case set is empty")
            phase = "test"
            if not _run_testing(
                logger,
                test_cases,
                isolation_tier,
                timing_metric,
                device_identity,
            ):
                return 0
            logger.raw(RESULT_SPLIT)
            phase = "benchmark"
            _run_benchmarking(
                logger,
                benchmark_cases,
                isolation_tier,
                timing_metric,
                device_identity,
            )
            return 0
        except BaseException as error:
            reason = _bounded_message(f"{type(error).__name__}: {error}")
            logger.log(
                "ferrl-infrastructure",
                f"v1 phase={phase} reason={json.dumps(reason, separators=(',', ':'))}",
            )
            return 114
    finally:
        _kill_candidate_tree()
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
