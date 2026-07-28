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
import io
import importlib
import multiprocessing
import os
import signal
import socket
import stat
import sys
import time


PR_SET_DUMPABLE = 4
PR_SET_CHILD_SUBREAPER = 36
SUBMISSION_PATH = "/opt/ferrl-verifier/submission.py"
RESULT_SPLIT = "===FERRL-BENCH==="
TIMING_METRIC = "isolated-service-latency-v1"
MAX_STATUS_BYTES = 256
MAX_STATUS_EVENTS = 32
MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
ATTEMPT_SENTINEL_PATH = "/work/cache/ferrl-attack-sentinel"
PARENT_DEVICE_CANARY = b"ferrl-parent-private-cuda-v1-7f4c3a19"
PAYLOAD_WIRE_PREFIX = b"FERRL-PAYLOAD-v1\0"


def _prctl(option, value):
    if ctypes.CDLL(None, use_errno=True).prctl(option, value, 0, 0, 0) != 0:
        raise SystemExit(114)


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

        def reject_import_protocol():
            _send_status(status, b"IMPORT_ERROR" if entered else b"IMPORT_REJECTED")

        while True:
            try:
                event, payload = _recv_payload(payload_results, MAX_STATUS_BYTES)
            except BaseException:
                reject_import_protocol()
                return
            if payload:
                reject_import_protocol()
                return
            if event == b"ENTRY":
                entered = True
                _send_status(status, b"ENTRY")
                continue
            if event == b"IMPORTED":
                _send_status(status, event)
                break
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
    def __init__(self, mode, logger):
        self.mode = mode
        self.logger = logger
        self.entered = False
        self.rejected = False
        self.closed = False
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
            if self._recv(before_entry=True) != b"READY":
                raise InfrastructureFailure("candidate worker trusted initialization failed")
            self.logger.log("ferrl-timing-metric", TIMING_METRIC)
            self.commands.send_bytes(b"ACK-v2")
            while True:
                event = self._recv(before_entry=not self.entered)
                if event == b"ENTRY":
                    self.entered = True
                    self.logger.log("ferrl-entry", f"{self.mode}-v4")
                    continue
                if event == b"IMPORTED" and self.entered:
                    break
                if event == b"IMPORT_REJECTED" and not self.entered:
                    self.rejected = True
                    self.logger.log("ferrl-candidate-rejected", f"{self.mode}-import-v1")
                    break
                if event == b"IMPORT_ERROR" and self.entered:
                    self.rejected = True
                    self.logger.log("ferrl-candidate-rejected", f"{self.mode}-import-v1")
                    break
                raise InfrastructureFailure("candidate worker import protocol failed")
        except CandidateFailure:
            if not self.entered:
                self.close()
                raise
            self.rejected = True
            self.logger.log("ferrl-candidate-rejected", f"{self.mode}-import-v1")
            self.close()
        except BaseException:
            self.close()
            raise

    def _recv(self, before_entry=False):
        for _ in range(MAX_STATUS_EVENTS):
            try:
                return self.status.recv_bytes(MAX_STATUS_BYTES)
            except (EOFError, OSError) as error:
                if before_entry:
                    raise InfrastructureFailure("candidate worker exited before entry") from error
                raise CandidateFailure("candidate worker exited after entry") from error
        raise CandidateFailure("candidate worker status flood")

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


def _run_testing(logger, tests):
    if not tests:
        raise InfrastructureFailure("trusted test case set is empty")
    session = CandidateSession("test", logger)
    try:
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


def _run_benchmarking(logger, tests):
    if not tests:
        raise InfrastructureFailure("trusted benchmark case set is empty")
    session = CandidateSession("benchmark", logger)
    try:
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
    seed = os.environ.pop("POPCORN_SEED", None)
    try:
        seed = int(seed) if seed else None
    except ValueError:
        return 114

    try:
        logger = GradeLogger(grade_socket)
    except BaseException:
        return 114

    try:
        phase = "preparation"
        try:
            _prctl(PR_SET_CHILD_SUBREAPER, 1)
            _SET_SEED(seed or 42)
            torch.cuda.init()
            _SYNC()
            test_cases = _GET_TEST_CASES(sys.argv[1], seed)
            benchmark_cases = _GET_TEST_CASES(sys.argv[2], seed)
            if not test_cases or not benchmark_cases:
                raise InfrastructureFailure("trusted case set is empty")
            phase = "test"
            if not _run_testing(logger, test_cases):
                return 0
            logger.raw(RESULT_SPLIT)
            phase = "benchmark"
            _run_benchmarking(logger, benchmark_cases)
            return 0
        except BaseException:
            logger.log("ferrl-infrastructure", f"v1 phase={phase}")
            return 114
    finally:
        _kill_candidate_tree()
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
