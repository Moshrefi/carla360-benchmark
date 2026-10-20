"""Compare complete unittest outcomes to the audited base; fail on new failures."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = "1a931e15881f583fd1c657faae7da8972835c91f"
RUNNER = """
import json, unittest
suite = unittest.defaultTestLoader.discover('tests')
result = unittest.TextTestRunner(verbosity=2).run(suite)
print('OUTCOMES_JSON=' + json.dumps({
    'tests': result.testsRun,
    'failed': sorted(t.id() for t, _ in result.failures + result.errors),
    'skipped': sorted(t.id() for t, _ in result.skipped)
}))
"""


def check(directory):
    p = subprocess.run([sys.executable, "-c", RUNNER], cwd=directory, text=True, capture_output=True)
    print(p.stderr)
    for line in p.stdout.splitlines():
        if line.startswith("OUTCOMES_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("Test runner did not produce an outcome report:\n" + p.stdout)


with tempfile.TemporaryDirectory() as temp:
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", BASE],
                             check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(temp, filter="data")
    baseline = check(temp)
    current = check(ROOT)
    report = {"audited_base": BASE, "baseline": baseline, "current": current,
              "new_failures": sorted(set(current["failed"]) - set(baseline["failed"]))}
    print(json.dumps(report, indent=2))
    if report["new_failures"]:
        raise SystemExit("New regression failures")
    if current["skipped"] != baseline["skipped"]:
        raise SystemExit("Unexpected change in skipped tests")
    if baseline["failed"]:
        print("Existing base failures remain explicitly reported; no new regression failures.")
