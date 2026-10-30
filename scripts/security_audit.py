"""
Security Audit Script for Agentic Chatbot repository.
Scans source code and configuration files for exposed secrets, passwords, private IPs, and keys.
"""

import os
import re
from pathlib import Path

# Sensitive patterns (generic matchers)
PATTERNS = [
    (re.compile(r"tvly-[A-Za-z0-9_-]{20,}", re.IGNORECASE), "Tavily API Key"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}", re.IGNORECASE), "OpenAI API Key"),
    (re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*[\"']([^\"']{4,})[\"']"), "Hardcoded Password Assignment"),
    (re.compile(r"192\.168\.\d+\.\d+"), "Private IP Address (192.168.x.x)"),
    (re.compile(r"(?i)alpha_vantage_api_key\s*[:=]\s*[\"'][A-Za-z0-9]{10,}[\"']"), "Hardcoded Alpha Vantage Key"),
]

EXCLUDED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".cache",
    "resources",
    "faiss_db",
    ".tmp",
}

EXCLUDED_FILES = {
    ".env",  # .env is expected to hold local secrets and will be gitignored
}


def run_audit(root_dir: Path):
    print("=" * 60)
    print("RUNNING SECURITY AUDIT")
    print(f"Directory: {root_dir.resolve()}")
    print("=" * 60)

    findings = []
    scanned_count = 0

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if (
            path.name in EXCLUDED_FILES
            or path.name.endswith(".pyc")
            or path.name.endswith(".db")
            or path.name.endswith(".bak")
        ):
            continue

        scanned_count += 1
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pattern, label in PATTERNS:
            for match in pattern.finditer(content):
                # Don't print the secret value, only the file, line, and category
                line_no = content[:match.start()].count("\n") + 1
                matched_text = match.group(0)
                # Allow dummy/placeholder matches
                if any(placeholder in matched_text.lower() for placeholder in ["your_", "example", "placeholder", "<password>", "test_dummy"]):
                    continue
                findings.append({
                    "file": str(path.relative_to(root_dir)),
                    "line": line_no,
                    "category": label,
                })

    print(f"Scanned {scanned_count} files.")
    if findings:
        print(f"\n[WARNING] Found {len(findings)} sensitive item(s) in codebase:")
        for f in findings:
            print(f"  - File: {f['file']} (Line {f['line']}): {f['category']}")
    else:
        print("\n[PASS] No exposed secrets, private IPs, or hardcoded passwords found in source files!")
    print("=" * 60)
    return findings


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    run_audit(repo_root)
