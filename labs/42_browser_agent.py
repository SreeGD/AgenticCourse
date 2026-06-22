"""Browser Automation — Session 42: Goal-Conditioned Browser Agent.

Full browser agent: accepts a natural-language goal, drives a Playwright
browser via the Computer Use API, and returns structured results.

Usage:
    python 42_browser_agent.py --goal "Extract the top 5 HN story titles and point counts"
    python 42_browser_agent.py --goal "..." --headless --output result.json
"""

import argparse
import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-7"
MAX_STEPS = 20
VIEWPORT = {"width": 1280, "height": 800}

SYSTEM_PROMPT = """You are a browser automation agent. You control a real web browser.

Rules:
- Always take a screenshot before deciding the next action when the page state is unclear
- If a page is loading (spinner, blank), wait and take another screenshot
- Extract data as structured JSON when the task is complete
- When the goal is fully achieved, respond with plain text (no tool use) summarising the result
- Never enter credentials unless they are explicitly provided in the goal
- If you cannot complete the goal after several attempts, explain why clearly
"""


async def take_screenshot(page) -> str:
    data = await page.screenshot()
    return base64.standard_b64encode(data).decode("utf-8")


def extract_tool_use(response: anthropic.types.Message) -> dict | None:
    for block in response.content:
        if block.type == "tool_use":
            return {"id": block.id, "name": block.name, "input": block.input}
    return None


async def execute_computer_action(page, inp: dict) -> str:
    action = inp.get("action", "")
    coord = inp.get("coordinate", [0, 0])

    match action:
        case "screenshot":
            return "screenshot captured"
        case "left_click":
            await page.mouse.click(coord[0], coord[1])
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
            return f"left_click at {coord}"
        case "right_click":
            await page.mouse.click(coord[0], coord[1], button="right")
            return f"right_click at {coord}"
        case "double_click":
            await page.mouse.dblclick(coord[0], coord[1])
            return f"double_click at {coord}"
        case "type":
            await page.keyboard.type(inp.get("text", ""))
            return f"typed: {inp.get('text', '')[:50]}"
        case "key":
            await page.keyboard.press(inp.get("key", ""))
            return f"key: {inp.get('key', '')}"
        case "scroll":
            direction = inp.get("direction", "down")
            amount = inp.get("amount", 3)
            delta_y = amount * 120 * (1 if direction == "down" else -1)
            await page.mouse.wheel(coord[0] if coord else 640, coord[1] if coord else 400, delta_y=delta_y)
            return f"scrolled {direction} by {amount}"
        case "mouse_move":
            await page.mouse.move(coord[0], coord[1])
            return f"moved mouse to {coord}"
        case _:
            return f"unhandled action: {action}"


async def run_agent(goal: str, headless: bool = True) -> dict:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {"error": "playwright not installed — run: pip install playwright && playwright install chromium"}

    client = anthropic.AsyncAnthropic()
    steps_log = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()
        await page.goto("about:blank")

        messages = []

        for step in range(1, MAX_STEPS + 1):
            screenshot_b64 = await take_screenshot(page)

            user_content = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64},
                },
                {"type": "text", "text": f"Goal: {goal}\n\nStep {step}/{MAX_STEPS}: What is the next action?"},
            ]
            messages.append({"role": "user", "content": user_content})

            response = await client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[
                    {
                        "type": "computer_20250124",
                        "name": "computer",
                        "display_width_px": VIEWPORT["width"],
                        "display_height_px": VIEWPORT["height"],
                    }
                ],
                messages=messages,
            )

            tool_use = extract_tool_use(response)

            if tool_use:
                action_result = await execute_computer_action(page, tool_use["input"])
                steps_log.append({"step": step, "action": tool_use["input"].get("action"), "result": action_result})
                print(f"  [{step}] {tool_use['input'].get('action')} → {action_result}")

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": action_result}],
                })
            else:
                # No tool use = task complete
                final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
                print(f"  [{step}] Task complete.")
                await browser.close()
                return {
                    "goal": goal,
                    "status": "completed",
                    "steps_taken": step,
                    "result": final_text,
                    "steps_log": steps_log,
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                }

            await asyncio.sleep(0.3)

        await browser.close()
        return {
            "goal": goal,
            "status": "max_steps_reached",
            "steps_taken": MAX_STEPS,
            "steps_log": steps_log,
        }


def main():
    parser = argparse.ArgumentParser(description="Goal-conditioned browser agent.")
    parser.add_argument("--goal", required=True, help="Natural language goal for the agent")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser headless")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode (visible)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    headless = not args.headed
    print(f"Goal: {args.goal}")
    print(f"Mode: {'headless' if headless else 'headed'}\n")

    result = asyncio.run(run_agent(args.goal, headless=headless))

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"\nResult written to {args.output}")
    else:
        print("\n" + output)


if __name__ == "__main__":
    main()
