"""Browser Automation — Session 42: Computer Use Demo.

Minimal demonstration of the screenshot-action loop using Anthropic's
Computer Use API. Shows the core pattern: take screenshot → ask Claude
→ execute action → repeat.

This demo runs a headless browser via Playwright, captures screenshots,
and feeds them to Claude with the Computer Use tool.

Usage:
    pip install anthropic playwright pillow
    playwright install chromium
    python 42_computer_use_demo.py
"""

import asyncio
import base64
import json
from io import BytesIO

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-7"

DEMO_GOAL = "Go to news.ycombinator.com and tell me the title of the #1 story."


async def take_screenshot_playwright(page) -> str:
    screenshot_bytes = await page.screenshot()
    return base64.standard_b64encode(screenshot_bytes).decode("utf-8")


def extract_action(response: anthropic.types.Message) -> dict | None:
    for block in response.content:
        if block.type == "tool_use" and block.name == "computer":
            return {"name": block.name, "input": block.input, "id": block.id}
    return None


async def execute_action(page, action: dict) -> str:
    inp = action["input"]
    action_type = inp.get("action")

    if action_type == "screenshot":
        return "screenshot taken"
    elif action_type == "left_click":
        await page.mouse.click(inp["coordinate"][0], inp["coordinate"][1])
        return f"clicked ({inp['coordinate'][0]}, {inp['coordinate'][1]})"
    elif action_type == "type":
        await page.keyboard.type(inp["text"])
        return f"typed: {inp['text'][:40]}"
    elif action_type == "key":
        await page.keyboard.press(inp["key"])
        return f"pressed: {inp['key']}"
    elif action_type == "scroll":
        direction = inp.get("direction", "down")
        amount = inp.get("amount", 3)
        delta = amount * 100 * (1 if direction == "down" else -1)
        await page.mouse.wheel(0, delta)
        return f"scrolled {direction}"
    elif action_type == "double_click":
        await page.mouse.dblclick(inp["coordinate"][0], inp["coordinate"][1])
        return f"double-clicked ({inp['coordinate'][0]}, {inp['coordinate'][1]})"
    else:
        return f"unknown action: {action_type}"


async def run_demo():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("playwright not installed. Run: pip install playwright && playwright install chromium")
        return

    client = anthropic.AsyncAnthropic()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        await page.goto("about:blank")

        messages = []
        max_steps = 10
        step = 0

        print(f"Goal: {DEMO_GOAL}\n")

        while step < max_steps:
            step += 1
            screenshot_b64 = await take_screenshot_playwright(page)

            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                },
                {
                    "type": "text",
                    "text": f"Goal: {DEMO_GOAL}\n\nStep {step}: What should I do next? If the goal is complete, explain the result and do not use any tools.",
                },
            ]
            messages.append({"role": "user", "content": user_content})

            response = await client.messages.create(
                model=MODEL,
                max_tokens=1024,
                tools=[
                    {
                        "type": "computer_20250124",
                        "name": "computer",
                        "display_width_px": 1280,
                        "display_height_px": 800,
                    }
                ],
                messages=messages,
            )

            print(f"Step {step} — stop_reason: {response.stop_reason}")

            action = extract_action(response)
            if action:
                result = await execute_action(page, action)
                print(f"  Action: {action['input'].get('action')} → {result}")

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": action["id"],
                            "content": result,
                        }
                    ],
                })
            else:
                # No tool use — Claude has finished
                for block in response.content:
                    if hasattr(block, "text"):
                        print(f"\nResult: {block.text}")
                break

            await asyncio.sleep(0.5)  # brief pause for page loads

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
