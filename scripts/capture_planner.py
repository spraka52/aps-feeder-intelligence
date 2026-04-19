"""Capture screenshots of the Planner view (clicks the role radio first)."""
from __future__ import annotations
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright


def capture(url: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(20)
        # Click the Planner radio
        try:
            page.get_by_text("Planner · week-by-week").click(timeout=10_000)
            print("clicked Planner radio")
        except Exception as e:
            print(f"failed to click Planner: {e}")
            return
        time.sleep(20)  # planner view needs time to compute weekly OpenDSS
        scroll_sel = 'section[data-testid="stMain"]'
        get_target = f"document.querySelector(`{scroll_sel}`)"
        max_scroll = page.evaluate(f"{get_target}.scrollHeight - {get_target}.clientHeight")
        print(f"max_scroll = {max_scroll}")
        # Pre-scroll once to mount everything
        for y in range(0, int(max_scroll) + 600, 600):
            page.evaluate(f"{get_target}.scrollTo(0, {y})")
            time.sleep(0.4)
        page.evaluate(f"{get_target}.scrollTo(0, 0)")
        time.sleep(2)
        max_scroll = page.evaluate(f"{get_target}.scrollHeight - {get_target}.clientHeight")
        print(f"updated max_scroll = {max_scroll}")
        n_panels = 4
        step = max_scroll / max(1, n_panels - 1)
        for i in range(n_panels):
            y = int(i * step)
            page.evaluate(f"{get_target}.scrollTo(0, {y})")
            time.sleep(2)
            out = out_dir / f"planner_{i+1}.png"
            page.screenshot(path=str(out), full_page=False)
            print(f"  -> {out}")
        b.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports/panels")
    capture(url, out_dir)
