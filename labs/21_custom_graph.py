"""Custom LangGraph + HITL — financial transaction approval workflow.

Drops down from `create_react_agent` to a custom `StateGraph`. Demonstrates:
  - Custom state schema (TypedDict)
  - Nodes as state-updating functions
  - Conditional edges (router function picks next node)
  - HITL via interrupt() — pauses execution, persists state, waits for resume
  - Command(resume=...) — how to feed the human decision back into the graph

Demo runs three scenarios:
  1. Small transaction ($50)         → auto-approve, no HITL
  2. Large transaction ($5000) APPROVED → HITL pauses, human approves, executes
  3. Large transaction ($5000) DENIED   → HITL pauses, human denies, blocks
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# =====================================================================
# State schema — every node reads from and updates this dict
# =====================================================================

class TxState(TypedDict, total=False):
    amount: float
    recipient: str
    reason: str
    status: str           # "proposed" | "approved" | "denied" | "executed" | "blocked"
    approver_note: str


# Threshold above which a transaction requires human approval
HITL_THRESHOLD = 1000.0


# =====================================================================
# Nodes — each is a function: state → partial state update
# =====================================================================

def propose(state: TxState) -> TxState:
    print(
        f"  [propose]      ${state['amount']:.2f} to {state['recipient']!r}\n"
        f"                  reason: {state.get('reason', '(none)')}"
    )
    return {"status": "proposed"}


def human_review(state: TxState) -> TxState:
    """
    Pauses the graph. The state passed to interrupt() is what the human sees
    (e.g., shown in a UI). The graph resumes when someone calls
    Command(resume=<decision>) from the outside.
    """
    print(f"  [human_review] ⏸  pausing for human approval...")
    decision = interrupt({
        "amount": state["amount"],
        "recipient": state["recipient"],
        "reason": state.get("reason", ""),
        "prompt": "Approve this transaction?",
    })
    # When resumed, `decision` is whatever was passed to Command(resume=...)
    return {
        "status": "approved" if decision.get("approved") else "denied",
        "approver_note": decision.get("note", ""),
    }


def execute(state: TxState) -> TxState:
    if state.get("status") == "denied":
        print(
            f"  [execute]      ❌ BLOCKED\n"
            f"                  approver_note: {state.get('approver_note', '(none)')!r}"
        )
        return {"status": "blocked"}

    print(f"  [execute]      ✅ EXECUTED ${state['amount']:.2f} → {state['recipient']!r}")
    return {"status": "executed"}


# =====================================================================
# Conditional edge router — returns the NAME of the next node
# =====================================================================

def needs_review(state: TxState) -> str:
    """Routes after `propose`: large transactions → human; small → straight to execute."""
    return "human_review" if state["amount"] > HITL_THRESHOLD else "execute"


# =====================================================================
# Build the graph
# =====================================================================

def build_graph():
    graph = StateGraph(TxState)
    graph.add_node("propose", propose)
    graph.add_node("human_review", human_review)
    graph.add_node("execute", execute)

    graph.add_edge(START, "propose")
    graph.add_conditional_edges("propose", needs_review, ["human_review", "execute"])
    graph.add_edge("human_review", "execute")
    graph.add_edge("execute", END)

    return graph.compile(checkpointer=MemorySaver())


# =====================================================================
# Demos
# =====================================================================

def run_scenario_1_small(agent) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO 1 — small transaction ($50): auto-approve, no HITL")
    print("=" * 70)
    config = {"configurable": {"thread_id": "tx-001"}}
    final = agent.invoke(
        {"amount": 50.0, "recipient": "Coffee Shop", "reason": "team coffee"},
        config=config,
    )
    print(f"\n  final state: status={final.get('status')!r}, amount=${final.get('amount')}")


def run_scenario_2_approved(agent) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO 2 — large transaction ($5000): HITL pauses, then APPROVED")
    print("=" * 70)
    config = {"configurable": {"thread_id": "tx-002"}}

    # Start the graph. It will run until the interrupt() in human_review.
    print("\n  starting graph...")
    interim = agent.invoke(
        {"amount": 5000.0, "recipient": "CRM Vendor Inc.", "reason": "Q3 annual license renewal"},
        config=config,
    )

    print(f"\n  graph paused. interim state shows: status={interim.get('status')!r}")
    print("  (state persisted in checkpointer; could resume from another process)")

    # Inspect what the human is being asked
    pending = agent.get_state(config)
    if pending.interrupts:
        request = pending.interrupts[0].value
        print(f"\n  human sees:")
        for k, v in request.items():
            print(f"    {k}: {v}")

    # Simulate human approval
    print("\n  ▶ human decision: APPROVED (resuming graph...)")
    final = agent.invoke(
        Command(resume={"approved": True, "note": "Renewal already budgeted"}),
        config=config,
    )
    print(f"\n  final state: status={final.get('status')!r}")


def run_scenario_3_denied(agent) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO 3 — large transaction ($5000): HITL pauses, then DENIED")
    print("=" * 70)
    config = {"configurable": {"thread_id": "tx-003"}}

    print("\n  starting graph...")
    agent.invoke(
        {"amount": 5000.0, "recipient": "Unknown Vendor", "reason": ""},
        config=config,
    )

    print("\n  ▶ human decision: DENIED (resuming graph with rejection...)")
    final = agent.invoke(
        Command(resume={"approved": False, "note": "Missing purchase order; unclear justification"}),
        config=config,
    )
    print(f"\n  final state: status={final.get('status')!r}")


if __name__ == "__main__":
    print("=" * 70)
    print("CUSTOM LangGraph — financial transaction approval with HITL")
    print("=" * 70)
    print(f"  HITL threshold: ${HITL_THRESHOLD:.2f}")

    agent = build_graph()

    run_scenario_1_small(agent)
    run_scenario_2_approved(agent)
    run_scenario_3_denied(agent)

    print("\n" + "=" * 70)
    print("WHAT JUST HAPPENED")
    print("=" * 70)
    print(
        "  • Three runs, three thread_ids, three independent state trajectories.\n"
        "  • Conditional edge (`needs_review`) routed by amount: small went\n"
        "    straight to execute; large took the human_review branch.\n"
        "  • interrupt() paused the graph mid-flow. State persisted in the\n"
        "    MemorySaver checkpointer. The 'resume' could have come from a\n"
        "    web UI, a Slack approval, or a second process minutes later.\n"
        "  • Command(resume=...) injected the human decision back into the\n"
        "    graph, which then routed deterministically to executed or blocked.\n"
        "\n"
        "  This is the canonical pattern for any HITL workflow — finance,\n"
        "  medical orders, content publishing, code deploys."
    )
