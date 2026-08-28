"""
scripts/check_pip_audit.py — Block CI only on *new*, unreviewed CVEs.

`pip-audit` alone is all-or-nothing: either it fails the build on every
known CVE in the dependency tree (impractical here — 65 entries, most
requiring a major-version bump across langgraph/langchain/streamlit/
starlette that needs its own dedicated upgrade pass) or it's silenced
entirely with `|| true`, which means it fails the build on nothing,
ever, including CVEs that show up tomorrow in a package nobody's
looking at.

This script runs pip-audit, diffs the result against a reviewed,
checked-in baseline (security/pip_audit_baseline.json), and exits
non-zero only for vulnerabilities that aren't already in that baseline.
Anything in the baseline is a vulnerability someone has actually looked
at and decided is acceptable for now — not a vulnerability nobody's
looked at.

Usage:
    python -m scripts.check_pip_audit                 # CI mode: fail on new CVEs
    python -m scripts.check_pip_audit --update-baseline # after reviewing new findings,
                                                          # accept everything currently
                                                          # found and rewrite the baseline
"""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_BASELINE_PATH = Path(__file__).parent.parent / "security" / "pip_audit_baseline.json"


def _run_pip_audit() -> list[dict]:
    """Run pip-audit and return a flat list of {package, id, fix_versions} dicts."""
    result = subprocess.run(
        ["pip-audit", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    # pip-audit exits 1 when it finds vulnerabilities — that's expected here,
    # we're parsing its output, not using its exit code directly.
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("pip-audit did not return valid JSON:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(2)

    found = []
    for dep in data.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            found.append({
                "package": dep["name"],
                "id": vuln["id"],
                "fix_versions": vuln.get("fix_versions", []),
            })
    return found


def _load_baseline() -> dict:
    if not _BASELINE_PATH.exists():
        return {"reviewed_date": "", "accepted_vulnerabilities": []}
    return json.loads(_BASELINE_PATH.read_text())


def _write_baseline(entries: list[dict]) -> None:
    entries = sorted(entries, key=lambda e: (e["package"], e["id"]))
    baseline = _load_baseline()
    baseline["reviewed_date"] = str(datetime.now(UTC).date())
    baseline["accepted_vulnerabilities"] = entries
    _BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")


def main() -> None:
    found = _run_pip_audit()
    baseline = _load_baseline()
    accepted = {(e["package"], e["id"]) for e in baseline["accepted_vulnerabilities"]}

    if "--update-baseline" in sys.argv:
        _write_baseline(found)
        print(f"Baseline updated with {len(found)} accepted vulnerabilities.")
        return

    new_findings = [f for f in found if (f["package"], f["id"]) not in accepted]

    print(f"pip-audit found {len(found)} known vulnerabilities.")
    print(f"{len(accepted)} are already in the reviewed baseline "
          f"(reviewed {baseline.get('reviewed_date', 'unknown')}).")

    if new_findings:
        print(f"\n{len(new_findings)} NEW vulnerability(ies) not in the baseline:\n")
        for f in new_findings:
            fix = ", ".join(f["fix_versions"]) or "no fix available"
            print(f"  {f['package']}: {f['id']} (fix: {fix})")
        print(
            "\nReview these. If accepted for now, run "
            "`python -m scripts.check_pip_audit --update-baseline` and commit "
            "the updated security/pip_audit_baseline.json with a reason in the PR."
        )
        sys.exit(1)

    print("No new vulnerabilities beyond the reviewed baseline.")


if __name__ == "__main__":
    main()
