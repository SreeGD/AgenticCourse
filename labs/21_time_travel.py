"""Time travel — replay a prior graph run from any checkpoint, with a different decision.

Every state change in a LangGraph run produces a checkpoint. The checkpointer
keeps the full history; you can `get_state_history()` to inspect it, then
re-invoke from any prior checkpoint to follow a different branch.

This demo:
  1. Runs the transaction-approval graph through approval (status=executed)
  2. Inspects get_state_history() — every checkpoint, in reverse order
  3. Rewinds to the checkpoint BEFORE human_review
  4. Resumes from there with DENIAL instead of approval
  5. Shows both outcomes from the same starting state

Production uses: debugging ('what would have happened if?'), counterfactual
audit ('show me the path where the human said no'), agent post-mortems.
"""

from langgraph.types import Command

# Import the graph builder from the sibling demo file.
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("custom_graph", HERE / "21_custom_graph.py")
custom_graph = importlib.util.module_from_spec(spec)
sys.modules["custom_graph"] = custom_graph
spec.loader.exec_module(custom_graph)


def show_history(agent, config: dict) -> None:
    """Pretty-print get_state_history() for inspection."""
    history = list(agent.get_state_history(config))
    print(f"\n  HISTORY (newest first, {len(history)} checkpoints):")
    for i, snap in enumerate(history):
        next_node = snap.next[0] if snap.next else "(end)"
        status = snap.values.get("status", "(none)")
        amount = snap.values.get("amount", "(none)")
        print(
            f"    [{i:>2}] next={next_node:<14} status={status:<10} "
            f"amount={amount}  checkpoint_id={snap.config['configurable']['checkpoint_id'][:8]}..."
        )


def find_checkpoint_before(agent, config: dict, target_node: str):
    """Return the first checkpoint (oldest forward, newest backward) whose `next` is `target_node`."""
    history = list(agent.get_state_history(config))
    for snap in history:
        if snap.next and snap.next[0] == target_node:
            return snap
    return None


if __name__ == "__main__":
    print("=" * 70)
    print("TIME TRAVEL — replay a graph from a prior checkpoint")
    print("=" * 70)

    agent = custom_graph.build_graph()
    config = {"configurable": {"thread_id": "tx-timetravel"}}

    # ─── Original timeline: approve the transaction ─────────────────────
    print("\n" + "=" * 70)
    print("STEP 1 — original timeline: large tx, APPROVED")
    print("=" * 70)

    agent.invoke(
        {"amount": 5000.0, "recipient": "CRM Vendor Inc.", "reason": "Q3 annual license renewal"},
        config=config,
    )
    final_original = agent.invoke(
        Command(resume={"approved": True, "note": "Renewal already budgeted"}),
        config=config,
    )
    print(f"  → original outcome: status={final_original.get('status')!r}")

    # ─── Inspect the history ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — inspect get_state_history()")
    print("=" * 70)
    show_history(agent, config)

    # ─── Find the checkpoint right before human_review ──────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — find the checkpoint BEFORE human_review")
    print("=" * 70)
    pre_review = find_checkpoint_before(agent, config, "human_review")
    if pre_review is None:
        raise RuntimeError("Could not find pre-human_review checkpoint")
    print(f"  found checkpoint: next={pre_review.next}, status={pre_review.values.get('status')!r}")
    print(f"  checkpoint_id: {pre_review.config['configurable']['checkpoint_id']}")

    # ─── Fork the timeline at that checkpoint with a DIFFERENT decision ──
    #
    # `Command(resume=...)` won't help here — once an interrupt has been
    # resolved in this thread, the resume value is sticky. The right pattern
    # for "fork from a checkpoint" is `update_state(as_node=...)` — we
    # manually inject what the node WOULD have returned, then let the graph
    # continue from there.
    print("\n" + "=" * 70)
    print("STEP 4 — fork at pre-review checkpoint, inject DENIAL via update_state")
    print("=" * 70)
    forked_config = agent.update_state(
        pre_review.config,
        {"status": "denied", "approver_note": "[time travel] reconsidered — denying instead"},
        as_node="human_review",   # pretend this update came from human_review
    )
    print(f"  forked checkpoint_id: {forked_config['configurable']['checkpoint_id'][:8]}...")

    print("\n  resuming from the forked checkpoint...")
    final_counterfactual = agent.invoke(None, config=forked_config)
    print(f"  → counterfactual outcome: status={final_counterfactual.get('status')!r}")

    # ─── Compare ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BEFORE/AFTER — same starting state, two outcomes")
    print("=" * 70)
    print(f"  original timeline:        status={final_original.get('status')!r}  (approved → executed)")
    print(f"  counterfactual timeline:  status={final_counterfactual.get('status')!r}  (denied → blocked)")
    print(f"  current approver_note:    {final_counterfactual.get('approver_note')!r}")

    print("\n" + "=" * 70)
    print("WHAT JUST HAPPENED")
    print("=" * 70)
    print(
        "  • Each step in a LangGraph run saves a checkpoint. The state\n"
        "    history is queryable and re-runnable from any prior point.\n"
        "  • Re-invoking from a prior checkpoint forks the timeline —\n"
        "    same initial state, different decisions, different outcomes.\n"
        "  • Production uses for time travel:\n"
        "      - Counterfactual debugging: 'what would the agent have done\n"
        "        if the user had said X instead of Y?'\n"
        "      - Compliance audits: 'show the path the system would have\n"
        "        followed under the rejected decision'\n"
        "      - Recovery: rewind to before a bad LLM call and replay with\n"
        "        a different prompt / model / tool\n"
        "  • Requires a checkpointer. MemorySaver is in-process only — for\n"
        "    durable time travel across restarts, use PostgresSaver."
    )
