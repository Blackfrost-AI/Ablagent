from pathlib import Path

import pytest

from ablagent.executors.local import LocalExecutor
from ablagent.executors.ssh import SSHExecutor
from ablagent.security.evidence import verify_evidence
from ablagent.security.secret_scan import scan_text


def test_secret_scan_detects_private_key():
    assert scan_text("-----BEGIN RSA PRIVATE KEY-----\n", path="fixture")


def test_secret_scan_detects_env_var_style_secrets():
    # Regression: keyword embedded in a longer env-var identifier.
    assert scan_text(
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", path="fixture"
    )
    assert scan_text("PASSWORD=hunter2hunter2hunter2", path="fixture")


def test_secret_scan_no_false_positive_on_prose():
    assert not scan_text("The secret to success is practice; refuse_rate: 1.0", path="fixture")


def test_evidence_digest_verification(tmp_path: Path):
    from ablagent.run import write_evidence

    evidence = write_evidence(tmp_path, stage="S0_INTAKE", kind="test", payload={"ok": True})
    path = next((tmp_path / "evidence").glob("*.json"))
    assert verify_evidence(path)["evidence_digest"] == evidence["evidence_digest"]
    path.write_text(path.read_text().replace("true", "false"))
    from ablagent.control.errors import IntegrityError

    with pytest.raises(IntegrityError):
        verify_evidence(path)


def test_local_executor_uses_argv(tmp_path: Path):
    result = LocalExecutor(["/bin/echo", "hello"]).run(stage="S0", inputs=[])
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"


def test_ssh_executor_does_not_build_shell_string():
    executor = SSHExecutor("example.invalid", ["echo", "hello; touch nope"])
    assert executor.argv == ("echo", "hello; touch nope")
