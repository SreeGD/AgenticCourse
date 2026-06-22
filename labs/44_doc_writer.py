"""Document & Slide Generation — Session 44: Context-Aware Document Generator.

Generates structured documents from raw input using document type templates.
Supports executive briefs, technical PRDs, sales proposals, and onboarding guides.

Usage:
    python 44_doc_writer.py --type executive_brief --input raw_notes.txt
    python 44_doc_writer.py --type technical_prd --input spec.txt --output prd.md
    python 44_doc_writer.py --list-types
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

MODEL = "claude-opus-4-7"
llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=4096)


DOC_TYPES = {
    "executive_brief": {
        "label": "Executive Brief",
        "audience": "C-suite / senior leadership",
        "tone": "direct, confident, no jargon",
        "length": "1 page, max 400 words",
        "structure": "Situation → Complication → Resolution → Ask (SCRA framework)",
        "notes": "Start with the ask or decision needed. Lead with outcomes, not process.",
    },
    "technical_prd": {
        "label": "Technical PRD (Product Requirements Document)",
        "audience": "engineering team",
        "tone": "precise, unambiguous, technical",
        "length": "3-5 pages",
        "structure": "Overview → Goals → Non-goals → User stories → Technical spec → Open questions → Success metrics",
        "notes": "Be explicit about what is OUT of scope. Every user story should be testable.",
    },
    "sales_proposal": {
        "label": "Sales Proposal",
        "audience": "client procurement / decision maker",
        "tone": "professional, benefit-focused, confident",
        "length": "2-3 pages",
        "structure": "Executive summary → Problem statement → Proposed solution → Why us → Pricing → Next steps",
        "notes": "Focus on client outcomes, not product features. Quantify ROI where possible.",
    },
    "onboarding_guide": {
        "label": "Onboarding Guide",
        "audience": "new team member / hire",
        "tone": "friendly, clear, actionable",
        "length": "2-4 pages",
        "structure": "Welcome → Context (why this role matters) → First week checklist → Key people → Resources → 30-60-90 day goals",
        "notes": "Use numbered checklists for action items. Include links to key resources.",
    },
    "postmortem": {
        "label": "Incident Postmortem",
        "audience": "engineering and operations teams",
        "tone": "blameless, factual, constructive",
        "length": "1-2 pages",
        "structure": "Incident summary → Timeline → Root cause → Impact → What went well → Action items",
        "notes": "Blameless tone throughout. Action items must have owners and due dates.",
    },
}

SYSTEM_PROMPT = """You are an expert technical writer producing professional documents.
Follow the document structure exactly. Match the tone for the specified audience.
Output clean markdown only — no preamble, no meta-commentary."""

DOC_PROMPT = """Document type: {label}
Target audience: {audience}
Tone: {tone}
Length: {length}
Required structure: {structure}
Notes: {notes}

Raw input to transform:
---
{raw_input}
---

Produce the complete document in markdown, following the structure exactly."""


def generate_document(doc_type: str, raw_input: str) -> str:
    if doc_type not in DOC_TYPES:
        raise ValueError(f"Unknown doc type '{doc_type}'. Use --list-types to see options.")

    spec = DOC_TYPES[doc_type]
    prompt = DOC_PROMPT.format(raw_input=raw_input, **spec)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Generate structured documents from raw input.")
    parser.add_argument("--type", dest="doc_type", help="Document type")
    parser.add_argument("--input", help="Raw input file (txt/md)")
    parser.add_argument("--output", default=None, help="Output markdown file")
    parser.add_argument("--list-types", action="store_true", help="List available document types")
    args = parser.parse_args()

    if args.list_types:
        print("Available document types:\n")
        for key, spec in DOC_TYPES.items():
            print(f"  {key:<20} — {spec['label']}")
        return

    if not args.doc_type or not args.input:
        parser.error("--type and --input are required unless using --list-types")

    raw_input = Path(args.input).read_text()
    print(f"Generating {DOC_TYPES[args.doc_type]['label']}...")

    document = generate_document(args.doc_type, raw_input)

    if args.output:
        Path(args.output).write_text(document)
        print(f"Document written to {args.output}")
    else:
        print("\n" + document)


if __name__ == "__main__":
    main()
