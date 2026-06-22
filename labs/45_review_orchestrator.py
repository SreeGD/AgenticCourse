"""Multi-Agent Code Review — Session 45: Orchestrator.

Fans out to four specialist reviewers in parallel, deduplicates findings,
sorts by severity, and optionally posts inline comments to a GitHub PR.

Usage:
    # Review a local diff
    git diff main...HEAD | python 45_review_orchestrator.py

    # Review a GitHub PR
    python 45_review_orchestrator.py --pr 123 --repo owner/repo

    # Review and post comments to GitHub
    python 45_review_orchestrator.py --pr 123 --repo owner/repo --post-comments

    # Save report to file
    python 45_review_orchestrator.py --pr 123 --repo owner/repo --output review.md
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

import reviewer_logic as logic_reviewer
import reviewer_docs as docs_reviewer
import reviewer_security as security_reviewer
import reviewer_style as style_reviewer
from review_schemas import Finding, FindingList, Severity, SEVERITY_RANK

load_dotenv()

MIN_CONFIDENCE = {
    Severity.CRITICAL: 0.70,
    Severity.MAJOR: 0.75,
    Severity.MINOR: 0.80,
    Severity.LOW: 0.85,
}


def fetch_pr_diff(pr_number: int, repo: str) -> str:
    result = subprocess.run(
        ["gh", "pr", "diff", str(pr_number), "--repo", repo],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Failed to fetch PR diff: {result.stderr}")
    return result.stdout


async def run_all_reviewers(diff: str) -> list[FindingList]:
    loop = asyncio.get_event_loop()

    results = await asyncio.gather(
        loop.run_in_executor(None, security_reviewer.review, diff),
        loop.run_in_executor(None, logic_reviewer.review, diff),
        loop.run_in_executor(None, style_reviewer.review, diff),
        loop.run_in_executor(None, docs_reviewer.review, diff),
        return_exceptions=True,
    )

    finding_lists = []
    labels = ["security", "logic", "style", "docs"]
    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            print(f"  [!] {label} reviewer failed: {result}", file=sys.stderr)
        else:
            count = len(result.findings)
            print(f"  ✓ {label:<10} → {count} finding{'s' if count != 1 else ''}")
            finding_lists.append(result)

    return finding_lists


def filter_by_confidence(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.confidence >= MIN_CONFIDENCE[f.severity]]


def deduplicate(findings: list[Finding]) -> list[Finding]:
    seen: dict[str, Finding] = {}
    for f in sorted(findings, key=lambda x: SEVERITY_RANK[x.severity], reverse=True):
        key = f"{f.file}:{f.line_start}-{f.line_end}"
        if key not in seen:
            seen[key] = f
        else:
            # Keep higher severity
            existing = seen[key]
            if SEVERITY_RANK[f.severity] > SEVERITY_RANK[existing.severity]:
                seen[key] = f
    return sorted(seen.values(), key=lambda x: SEVERITY_RANK[x.severity], reverse=True)


def format_markdown_report(findings: list[Finding]) -> str:
    if not findings:
        return "## Code Review\n\nNo issues found. ✓\n"

    counts = {s: sum(1 for f in findings if f.severity == s) for s in Severity}
    lines = [
        "## Code Review Summary\n",
        f"| Severity | Count |",
        f"|----------|-------|",
        f"| CRITICAL | {counts[Severity.CRITICAL]} |",
        f"| MAJOR    | {counts[Severity.MAJOR]} |",
        f"| MINOR    | {counts[Severity.MINOR]} |",
        f"| LOW      | {counts[Severity.LOW]} |",
        f"| **Total**| **{len(findings)}** |",
        "",
        "## Findings\n",
    ]

    for f in findings:
        lines += [
            f"### [{f.severity}] {f.title}",
            f"**File:** `{f.file}` (lines {f.line_start}–{f.line_end})  ",
            f"**Category:** {f.category}  ",
            f"**Confidence:** {f.confidence:.0%}\n",
            f"{f.description}\n",
            f"**Suggestion:** {f.suggestion}\n",
            "---",
        ]

    return "\n".join(lines)


def post_pr_comments(pr_number: int, repo: str, findings: list[Finding]):
    summary = format_markdown_report(findings)
    has_critical = any(f.severity == Severity.CRITICAL for f in findings)

    review_cmd = [
        "gh", "pr", "review", str(pr_number),
        "--repo", repo,
        "--request-changes" if has_critical else "--comment",
        "--body", summary,
    ]
    result = subprocess.run(review_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to post review: {result.stderr}", file=sys.stderr)
    else:
        print(f"Posted review to PR #{pr_number} ({'requesting changes' if has_critical else 'comment'})")


async def main():
    parser = argparse.ArgumentParser(description="Multi-agent code review orchestrator.")
    parser.add_argument("--pr", type=int, default=None, help="GitHub PR number")
    parser.add_argument("--repo", default=None, help="GitHub repo (owner/repo)")
    parser.add_argument("--post-comments", action="store_true", help="Post findings as PR comments")
    parser.add_argument("--output", default=None, help="Save markdown report to file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of markdown")
    args = parser.parse_args()

    # Get the diff
    if args.pr and args.repo:
        print(f"Fetching diff for PR #{args.pr} ({args.repo})...")
        diff = fetch_pr_diff(args.pr, args.repo)
    elif not sys.stdin.isatty():
        diff = sys.stdin.read()
    else:
        parser.error("Provide --pr and --repo, or pipe a diff via stdin.")

    if not diff.strip():
        sys.exit("Empty diff — nothing to review.")

    lines_changed = diff.count("\n+") + diff.count("\n-")
    print(f"Running 4 specialist reviewers on {lines_changed} changed lines...\n")

    finding_lists = await run_all_reviewers(diff)

    all_findings = [f for fl in finding_lists for f in fl.findings]
    confident = filter_by_confidence(all_findings)
    unique = deduplicate(confident)

    print(f"\nDeduplication: {len(all_findings)} raw → {len(confident)} confident → {len(unique)} unique")
    print(f"\nSeverity breakdown:")
    for sev in Severity:
        count = sum(1 for f in unique if f.severity == sev)
        if count:
            print(f"  {sev.value:<10} {count}")

    if args.json:
        output = json.dumps({"findings": [f.model_dump() for f in unique]}, indent=2)
    else:
        output = format_markdown_report(unique)

    if args.output:
        Path(args.output).write_text(output)
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + output)

    if args.post_comments and args.pr and args.repo:
        post_pr_comments(args.pr, args.repo, unique)


if __name__ == "__main__":
    asyncio.run(main())
