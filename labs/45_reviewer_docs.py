"""Multi-Agent Code Review — Session 45: Documentation Specialist Reviewer."""

import json
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from review_schemas import Finding, FindingList

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"  # Docs review is lightweight — Haiku is sufficient
llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=2048)

SYSTEM = """You are a documentation reviewer. Your ONLY job is finding documentation gaps.

Look for:
- Public functions/classes/modules missing docstrings
- Docstrings that are outdated (describe old behavior no longer present)
- Missing parameter documentation for complex or non-obvious parameters
- Missing return value documentation
- TODO/FIXME comments that should be resolved or tracked as issues
- Changelog / migration notes missing for breaking changes
- Missing example usage for complex APIs
- Code comments that explain WHAT instead of WHY (comments should explain intent)

Severity:
- CRITICAL: not applicable for docs (never use)
- MAJOR: public API missing docs, making it unusable without reading the source
- MINOR: internal function missing docstring or partially documented
- LOW: minor comment quality issue

Only report findings with confidence >= 0.85.
Return ONLY valid JSON. No preamble."""

PROMPT = """Review this code diff for documentation issues:

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
      "severity": "LOW",
      "category": "docs",
      "title": "Short title",
      "description": "What documentation is missing or wrong",
      "suggestion": "What to add or fix",
      "confidence": 0.9
    }}
  ]
}}

If no doc issues found, return {{"findings": []}}"""


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
