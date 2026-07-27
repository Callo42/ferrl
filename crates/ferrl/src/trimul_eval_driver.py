"""Ferrl-owned TriMul verifier boundary.

The protected parent owns input generation, correctness checking, elapsed-time
measurement, statistics, and the machine-grade socket. Candidate Python executes
only in a spawned child process. The child can corrupt its own interpreter and
control pipes, but it cannot mutate the checker/timer process or write the grade.
Every reported output is copied into a fresh parent-owned CUDA buffer and checked
on a fresh hidden-seed input before a duration is accepted.
"""

import ctypes
import importlib
import multiprocessing
import os
import signal
import socket
import sys
import time


PR_SET_DUMPABLE = 4
PR_SET_CHILD_SUBREAPER = 36
SUBMISSION_PATH = "/opt/ferrl-verifier/submission.py"
RESULT_SPLIT = "===FERRL-BENCH==="
MAX_STATUS_BYTES = 256
MAX_STATUS_EVENTS = 32


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


def _send_status(connection, value):
    try:
        connection.send_bytes(value)
    except BaseException:
        os._exit(115)


def _candidate_worker(commands, status):
    _prctl(PR_SET_DUMPABLE, 0)
    try:
        os.setsid()
    except OSError:
        pass

    # Capture exact trusted callables before candidate import can replace module
    # globals in this disposable process.
    sync = _SYNC
    tensor_type = _TENSOR_TYPE
    copy_tensor = _COPY_TENSOR
    try:
        torch.cuda.init()
        sync()
    except BaseException:
        _send_status(status, b"TRUSTED_INIT_ERROR")
        return
    _send_status(status, b"READY")

    try:
        if commands.recv() != ("ack-v1",):
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
            _send_status(status, b"ENTRY")
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
        _send_status(status, b"IMPORT_ERROR" if entry_sent else b"IMPORT_REJECTED")
        return
    if not entry_sent:
        _send_status(status, b"IMPORT_REJECTED")
        return
    _send_status(status, b"IMPORTED")

    candidate_data = None
    output_buffer = None
    while True:
        try:
            command = commands.recv()
        except EOFError:
            return
        except BaseException:
            _send_status(status, b"CONTROL_ERROR")
            return
        if command == ("close-v1",):
            return
        if not isinstance(command, tuple) or not command:
            _send_status(status, b"CONTROL_ERROR")
            return
        if command[0] == "prepare-v1" and len(command) == 3:
            candidate_data = command[1]
            output_buffer = command[2]
            _send_status(status, b"PREPARED")
            continue
        if command != ("run-v1",) or candidate_data is None or output_buffer is None:
            _send_status(status, b"CONTROL_ERROR")
            return
        try:
            output = kernel(candidate_data)
            if type(output) is not tensor_type:
                raise TypeError("candidate output is not an exact torch.Tensor")
            if (
                output.shape != output_buffer.shape
                or output.dtype != output_buffer.dtype
                or output.device != output_buffer.device
            ):
                raise ValueError("candidate output schema mismatch")
            copy_tensor(output_buffer, output)
            sync()
        except BaseException:
            _send_status(status, b"CANDIDATE_ERROR")
            continue
        _send_status(status, b"DONE")


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
        self.worker = context.Process(
            target=_candidate_worker,
            args=(child_commands, child_status),
            daemon=False,
        )
        self.worker.start()
        child_commands.close()
        child_status.close()
        try:
            if self._recv(before_entry=True) != b"READY":
                raise InfrastructureFailure("candidate worker trusted initialization failed")
            self.commands.send(("ack-v1",))
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

    def execute(self, candidate_data, shared_output, checked_output):
        if self.rejected:
            raise CandidateFailure("candidate import was rejected")
        # The candidate must not see either the hidden-seed input or its writable
        # result buffer before the protected timer starts. Candidate code can
        # corrupt every object in its own interpreter, including the worker's
        # control connection, so a DONE event is only a prompt for the protected
        # parent to capture and synchronize a fresh private output.
        started = time.perf_counter_ns()
        try:
            self.commands.send(("prepare-v1", candidate_data, shared_output))
        except BaseException as error:
            raise CandidateFailure("candidate input handoff failed") from error
        if self._recv() != b"PREPARED":
            raise CandidateFailure("candidate prepare protocol failed")
        try:
            self.commands.send(("run-v1",))
        except BaseException as error:
            raise CandidateFailure("candidate run handoff failed") from error
        status = self._recv()
        if status != b"DONE":
            raise CandidateFailure("candidate execution failed")
        try:
            _COPY_TENSOR(checked_output, shared_output)
            _SYNC()
        except BaseException as error:
            raise InfrastructureFailure("trusted output capture failed") from error
        return time.perf_counter_ns() - started

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.worker.is_alive():
                self.commands.send(("close-v1",))
        except BaseException:
            pass
        self.worker.join(0.25)
        if self.worker.is_alive():
            self.worker.kill()
            self.worker.join(1)
        self.commands.close()
        self.status.close()
        _kill_candidate_tree()


def _wrap_check(data, output):
    result = _CHECK_IMPLEMENTATION(data, output)
    if isinstance(result, tuple):
        return result
    return not bool(result), result


def _execute_checked(session, args):
    try:
        data = _GENERATE_INPUT(**args)
        candidate_data = _CLONE_DATA(data)
        shared_output = torch.full_like(data[0], float("nan"))
        checked_output = torch.full_like(data[0], float("nan"))
    except BaseException as error:
        raise InfrastructureFailure("trusted input/output preparation failed") from error
    elapsed = session.execute(candidate_data, shared_output, checked_output)
    try:
        good, message = _wrap_check(data, checked_output)
    except BaseException as error:
        raise InfrastructureFailure("trusted correctness checker failed") from error
    return bool(good), _bounded_message(message or ""), elapsed


def _run_testing(logger, tests):
    session = CandidateSession("test", logger)
    try:
        logger.log("test-count", len(tests))
        if session.rejected:
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
        _prctl(PR_SET_CHILD_SUBREAPER, 1)
        _SET_SEED(seed or 42)
        torch.cuda.init()
        _SYNC()
        test_cases = _GET_TEST_CASES(sys.argv[1], seed)
        benchmark_cases = _GET_TEST_CASES(sys.argv[2], seed)
        logger = GradeLogger(grade_socket)
    except BaseException:
        return 114

    try:
        try:
            if not _run_testing(logger, test_cases):
                return 0
            logger.raw(RESULT_SPLIT)
            _run_benchmarking(logger, benchmark_cases)
            return 0
        except InfrastructureFailure:
            logger.log("ferrl-infrastructure", "v1")
            return 114
    finally:
        _kill_candidate_tree()
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
