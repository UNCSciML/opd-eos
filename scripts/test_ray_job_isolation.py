import os
from pathlib import Path
import subprocess
import time
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "ray_job_isolation.sh"
LAUNCHER = REPO_ROOT / "run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"


def _marker_process(marker: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PROCESS_MARKER"] = marker
    return subprocess.Popen(
        ["bash", "-c", 'exec -a "$PROCESS_MARKER" sleep 60'],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_cleanup_only_stops_processes_from_current_ray_tmpdir():
    unique = uuid.uuid4().hex
    current_marker = f"/tmp/opd_sampled_token_test_{unique}_current"
    other_marker = f"/tmp/opd_sampled_token_test_{unique}_other"
    current = _marker_process(current_marker)
    other = _marker_process(other_marker)
    try:
        time.sleep(0.1)
        env = os.environ.copy()
        env["RAY_TMPDIR"] = current_marker
        result = subprocess.run(
            ["bash", "-c", f'source "{HELPER}"; opd_cleanup_ray_session'],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        current.wait(timeout=2)
        assert other.poll() is None
    finally:
        for process in (current, other):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)


def test_export_ray_address_uses_current_job_cluster_file(tmp_path):
    (tmp_path / "ray_current_cluster").write_text("127.0.0.1:43210\n")
    env = os.environ.copy()
    env["RAY_TMPDIR"] = str(tmp_path)

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{HELPER}"; opd_export_ray_address; printf "%s" "$RAY_ADDRESS"',
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "127.0.0.1:43210"


def test_launcher_accepts_an_isolated_ray_worker_port_range():
    env = os.environ.copy()
    env.update(
        DRY_RUN="true",
        PROJECT_ROOT=str(REPO_ROOT),
        RAY_MIN_WORKER_PORT="20000",
        RAY_MAX_WORKER_PORT="24999",
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ray_worker_port_range=20000-24999" in result.stdout
