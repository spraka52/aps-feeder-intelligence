"""Capture multiple viewport screenshots while scrolling the Streamlit app.

Streamlit's main pane uses a virtualized scrollable container, so playwright's
full_page mode often misses the lower sections. We instead take a series of
viewport snapshots and write them as panel_1.png, panel_2.png, ...
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def capture_panels(url: str, out_dir: Path, n_panels: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    viewport = {"width": 1500, "height": 1000}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=viewport, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=60_000)
        time.sleep(12)
        try:
            page.wait_for_selector("text=Operator KPIs", timeout=10_000)
        except Exception:
            pass
        # Streamlit's main pane is the scroll container.
        scroll_sel = 'section[data-testid="stMain"]'
        page.wait_for_selector(scroll_sel, timeout=10_000)
        get_target_js = f"document.querySelector(`{scroll_sel}`)"
        max_scroll = page.evaluate(f"{get_target_js}.scrollHeight - {get_target_js}.clientHeight")
        print(f"max_scroll (stMain) = {max_scroll}px")
        # Pre-scroll once to make sure every chart mounts.
        for y in range(0, int(max_scroll) + 600, 500):
            page.evaluate(f"{get_target_js}.scrollTo(0, {y})")
            time.sleep(0.35)
        try:
            page.get_by_text("Model performance (held-out validation)").click(timeout=2000)
            time.sleep(1.5)
        except Exception:
            pass
        page.evaluate(f"document.querySelector('{scroll_sel}').scrollTo(0, 0)")
        time.sleep(1.5)
        max_scroll = page.evaluate(
            f"document.querySelector('{scroll_sel}').scrollHeight - document.querySelector('{scroll_sel}').clientHeight"
        )
        print(f"updated max_scroll = {max_scroll}px")
        step = max_scroll / max(1, n_panels - 1)
        for i in range(n_panels):
            y = int(i * step)
            page.evaluate(f"{get_target_js}.scrollTo(0, {y})")
            time.sleep(2.0)
            out = out_dir / f"panel_{i+1}.png"
            page.screenshot(path=str(out), full_page=False)
            print(f"  -> {out}  (scroll y={y})")
        browser.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports/panels")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    capture_panels(url, out_dir, n)
