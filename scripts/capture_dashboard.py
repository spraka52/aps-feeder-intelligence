"""Headless full-page screenshot of the running Streamlit dashboard."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def capture(url: str, out_path: Path, wait_seconds: int = 12, viewport_w: int = 1600, viewport_h: int = 1100):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": viewport_w, "height": viewport_h}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=60_000)
        # Streamlit re-renders after first paint; give it time to finish solving + plotting
        time.sleep(wait_seconds)
        try:
            page.wait_for_selector("text=Operator KPIs", timeout=10_000)
        except Exception:
            pass
        # Streamlit lazy-renders Plotly charts as they enter the viewport. Scroll
        # all the way down so every chart mounts, then back up.
        for y in range(0, 8000, 600):
            page.evaluate(f"window.scrollTo(0, {y})")
            time.sleep(0.4)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        # Expand the "Model performance" expander so its content shows up
        try:
            page.get_by_text("Model performance (held-out validation)").click(timeout=2000)
            time.sleep(1)
        except Exception:
            pass
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)
        try:
            page.wait_for_function("document.querySelectorAll('div.js-plotly-plot').length >= 6", timeout=15_000)
        except Exception:
            pass
        time.sleep(2)
        page.screenshot(path=str(out_path), full_page=True)
        browser.close()
    return out_path


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports/dashboard_full.png")
    p = capture(url, out)
    print(f"Saved: {p}  ({p.stat().st_size // 1024} KB)")
