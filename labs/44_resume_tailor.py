"""Document & Slide Generation — Session 44: ATS Resume Tailor.

Takes a job description and a raw resume (markdown/text), extracts
JD keywords, scores the resume, rewrites it for ATS compliance, and
exports a clean markdown file (optionally converting to PDF via pandoc).

Usage:
    python 44_resume_tailor.py --jd job_description.txt --resume my_resume.md
    python 44_resume_tailor.py --jd job.txt --resume resume.md --output tailored.md
"""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "claude-opus-4-7"
llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=4096)


SCORE_SYSTEM = """You are an ATS (Applicant Tracking System) expert.
Analyse the job description and resume, then return a JSON object."""

SCORE_PROMPT = """Job Description:
{jd}

---

Resume:
{resume}

---

Return JSON with this exact structure:
{{
  "top_keywords": ["keyword1", "keyword2", ...],   // top 15 keywords/skills from JD
  "found_in_resume": ["kw1", ...],                 // which of top_keywords appear in resume
  "missing": ["kw1", ...],                         // which are absent
  "ats_score": 72,                                  // 0-100 match score
  "priority_missing": ["kw1", "kw2", "kw3", "kw4", "kw5"]  // top 5 to add
}}"""


REWRITE_SYSTEM = """You are an expert resume writer specialising in ATS optimisation.

Rules:
- Naturally incorporate the priority keywords into the resume
- Keep ALL factual claims true — never fabricate experience or dates
- Use standard section headers: Summary, Experience, Skills, Education
- Avoid tables, columns, headers/footers, images — ATS parsers cannot handle them
- Use strong action verbs aligned with the JD language
- Quantify achievements where the original has numbers
- Output clean markdown only — no preamble, no explanation"""

REWRITE_PROMPT = """Job Description:
{jd}

---

Original Resume:
{resume}

---

Priority keywords to include: {priority_keywords}

Produce the ATS-optimised resume in markdown."""


class KeywordScore(BaseModel):
    top_keywords: list[str]
    found_in_resume: list[str]
    missing: list[str]
    ats_score: int = Field(ge=0, le=100)
    priority_missing: list[str]


def score_resume(jd: str, resume: str) -> KeywordScore:
    response = llm.invoke([
        SystemMessage(content=SCORE_SYSTEM),
        HumanMessage(content=SCORE_PROMPT.format(jd=jd, resume=resume)),
    ])
    raw = response.content
    # Extract JSON from the response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return KeywordScore(**json.loads(raw[start:end]))


def rewrite_resume(jd: str, resume: str, priority_keywords: list[str]) -> str:
    response = llm.invoke([
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=REWRITE_PROMPT.format(
            jd=jd,
            resume=resume,
            priority_keywords=", ".join(priority_keywords),
        )),
    ])
    return response.content


def export_pdf(markdown_path: Path) -> Path | None:
    pdf_path = markdown_path.with_suffix(".pdf")
    result = subprocess.run(
        ["pandoc", str(markdown_path), "-o", str(pdf_path)],
        capture_output=True
    )
    return pdf_path if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description="ATS-optimised resume tailor.")
    parser.add_argument("--jd", required=True, help="Job description file (txt/md)")
    parser.add_argument("--resume", required=True, help="Your resume file (txt/md)")
    parser.add_argument("--output", default=None, help="Output markdown file")
    parser.add_argument("--pdf", action="store_true", help="Export PDF via pandoc")
    parser.add_argument("--score-only", action="store_true", help="Only score, do not rewrite")
    args = parser.parse_args()

    jd = Path(args.jd).read_text()
    resume = Path(args.resume).read_text()

    print("Scoring current resume against JD...")
    score = score_resume(jd, resume)

    print(f"\n  ATS score:        {score.ats_score}/100")
    print(f"  Top JD keywords:  {', '.join(score.top_keywords[:8])}...")
    print(f"  Found in resume:  {len(score.found_in_resume)}/{len(score.top_keywords)}")
    print(f"  Missing priority: {', '.join(score.priority_missing)}")

    if args.score_only:
        return

    print("\nRewriting resume for ATS...")
    tailored = rewrite_resume(jd, resume, score.priority_missing)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        stem = Path(args.resume).stem
        out_path = Path(f"{stem}_tailored_{date.today().isoformat()}.md")

    out_path.write_text(tailored)
    print(f"\nTailored resume written to {out_path}")

    if args.pdf:
        pdf = export_pdf(out_path)
        if pdf:
            print(f"PDF exported to {pdf}")
        else:
            print("PDF export failed — is pandoc installed? (brew install pandoc)")

    print("\n--- Preview (first 300 chars) ---")
    print(tailored[:300] + "...")


if __name__ == "__main__":
    main()
