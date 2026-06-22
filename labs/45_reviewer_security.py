"""Multi-Agent Code Review — Session 45: Security Specialist Reviewer.

Invoked by 45_review_orchestrator.py. Reviews a code diff exclusively
for security vulnerabilities and returns structured findings.
"""

import json
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from review_schemas import Finding, FindingList, Severity

load_dotenv()

MODEL = "claude-opus-4-7"
llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=4096)

SYSTEM = """You are a security-focused code reviewer. Your ONLY job is finding security vulnerabilities.

Look for:
- SQL injection (string concatenation into queries, f-strings in SQL)
- Command injection (user input passed to shell commands, subprocess, os.system)
- XSS (unescaped user output rendered in HTML)
- Insecure deserialization (pickle, yaml.load without Loader)
- Hardcoded secrets, API keys, or credentials in code
- Missing authentication or authorization checks on endpoints
- Path traversal (user-controlled file paths without sanitisation)
- IDOR (Insecure Direct Object References — access by ID without ownership check)
- SSRF (user-controlled URLs fetched server-side)
- Missing input validation at API boundaries
- Insecure use of cryptography (MD5, SHA1 for passwords, weak random)
- Open redirects

Severity classification:
- CRITICAL: remotely exploitable without auth, data exfiltration risk, RCE
- MAJOR: requires auth or chaining; genuinely exploitable
- MINOR: defense-in-depth gap; not directly exploitable
- LOW: best practice violation; informational

Only report findings with confidence >= 0.7.
Return ONLY valid JSON matching the FindingList schema. No preamble."""

PROMPT = """Review this code diff for security vulnerabilities:

```diff
{diff}
```

Return JSON:
{{
  "findings": [
    {{
      "file": "path/to/file.py",
      "line_start": 42,
      "line_end": 45,
      "severity": "CRITICAL",
      "category": "security",
      "title": "Short title",
      "description": "Detailed explanation of the vulnerability",
      "suggestion": "Concrete fix",
      "confidence": 0.95
    }}
  ]
}}

If no security issues found, return {{"findings": []}}"""


def review(diff: str) -> FindingList:
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=PROMPT.format(diff=diff)),
    ])
    raw = response.content
    start = raw.find("{")
    end = raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    return FindingList(findings=[Finding(**f) for f in data.get("findings", [])])


if __name__ == "__main__":
    diff = sys.stdin.read()
    result = review(diff)
    print(json.dumps({"findings": [f.model_dump() for f in result.findings]}, indent=2))
