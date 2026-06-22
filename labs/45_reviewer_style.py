"""Multi-Agent Code Review — Session 45: Style & Maintainability Specialist Reviewer."""

import json
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from review_schemas import Finding, FindingList

load_dotenv()

MODEL = "claude-sonnet-4-6"  # Style review doesn't need Opus
llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=4096)

SYSTEM = """You are a code style and maintainability reviewer. Your ONLY job is finding style and readability issues.

Look for:
- Misleading or unclear variable/function names
- Functions that do more than one thing (violate single responsibility)
- Excessive complexity (deeply nested conditionals, long functions >50 lines)
- Code duplication that should be extracted (3+ similar blocks)
- Magic numbers/strings that should be named constants
- Dead code (unreachable branches, unused variables/imports)
- Inconsistent naming conventions within the file
- Missing type hints on public functions
- Overly clever code that sacrifices readability
- Long parameter lists (>5 params — consider a dataclass/config object)

Severity:
- CRITICAL: not applicable for style (never use)
- MAJOR: significantly harms readability for the whole team
- MINOR: noticeable but contained style issue
- LOW: minor preference or convention gap

Only report findings with confidence >= 0.8 (be conservative — style is subjective).
Return ONLY valid JSON. No preamble."""

PROMPT = """Review this code diff for style and maintainability issues:

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
      "severity": "MINOR",
      "category": "style",
      "title": "Short title",
      "description": "Explanation of the style issue",
      "suggestion": "Concrete improvement",
      "confidence": 0.85
    }}
  ]
}}

If no style issues found, return {{"findings": []}}"""


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
