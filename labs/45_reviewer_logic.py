"""Multi-Agent Code Review — Session 45: Logic & Correctness Specialist Reviewer."""

import json
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from review_schemas import Finding, FindingList

load_dotenv()

MODEL = "claude-opus-4-7"
llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=4096)

SYSTEM = """You are a logic and correctness code reviewer. Your ONLY job is finding bugs and incorrect behavior.

Look for:
- Off-by-one errors (loop bounds, slice indices, pagination)
- Null/None dereferences without guards
- Race conditions in concurrent or async code
- Integer overflow / underflow
- Incorrect error handling (swallowed exceptions, wrong HTTP status codes)
- Logical contradictions (conditions that can never be true or always true)
- Missing edge case handling (empty list, zero, empty string, None input)
- Incorrect algorithm implementation (sorting, searching, hashing)
- State machine violations (calling methods in wrong order)
- API contract violations (wrong argument types, missing required args)
- Division by zero without guard
- Infinite loops or missing termination conditions

Severity:
- CRITICAL: causes data loss, corruption, or system crash
- MAJOR: causes incorrect behavior for common/expected inputs
- MINOR: causes incorrect behavior only for edge cases
- LOW: suboptimal but technically correct

Only report findings with confidence >= 0.7.
Return ONLY valid JSON. No preamble."""

PROMPT = """Review this code diff for logic and correctness bugs:

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
      "severity": "MAJOR",
      "category": "logic",
      "title": "Short title",
      "description": "Detailed explanation of the bug",
      "suggestion": "Concrete fix",
      "confidence": 0.9
    }}
  ]
}}

If no logic bugs found, return {{"findings": []}}"""


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
