"""Export presentation assets from the executed notebook.

Pulls the embedded plot PNGs and the final comparison table out of
``notebook.ipynb`` into ``presentation/`` (PNGs + CSV + a rendered table image).
Run after executing the notebook:  uv run python scripts/export_presentation.py
"""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
OUT.mkdir(exist_ok=True)

# Plot title substring -> output filename.
PLOTS = {
    "Wide universe vs benchmarks": "equity_curves.png",
    "MA 7/21 strategy drawdown": "drawdown.png",
    "Portfolio weight allocations by coin": "portfolio_weights.png",
    "Dynamic weight evolution": "weight_evolution.png",
}

nb = nbformat.read(ROOT / "notebook.ipynb", as_version=4)


def save_plots() -> None:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for title, fname in PLOTS.items():
            if title in cell.source:
                for out in cell.get("outputs", []):
                    png = out.get("data", {}).get("image/png")
                    if png:
                        (OUT / fname).write_bytes(base64.b64decode(png))
                        print(f"saved {fname}")


def save_table() -> pd.DataFrame:
    html = None
    for cell in nb.cells:
        if cell.cell_type == "code" and "all_strategies = {" in cell.source:
            for out in cell.get("outputs", []):
                html = out.get("data", {}).get("text/html")
    if not html:
        raise SystemExit("final table HTML not found — execute the notebook first")

    table = BeautifulSoup(html, "html.parser").find("table")
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    headers[0] = "strategy"
    rows = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            for tr in table.find("tbody").find_all("tr")]
    df = pd.DataFrame(rows, columns=headers).set_index("strategy")
    df.to_csv(OUT / "final_comparison.csv")
    print("saved final_comparison.csv")

    # Render the table as an image for slides.
    fig, ax = plt.subplots(figsize=(17, 5))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title("All strategies — final comparison (net of costs, 365-annualized)", pad=20)
    fig.tight_layout()
    fig.savefig(OUT / "final_comparison.png", dpi=150, bbox_inches="tight")
    print("saved final_comparison.png")
    return df


if __name__ == "__main__":
    save_plots()
    df = save_table()
    print(f"\n{len(df)} strategies exported to {OUT}/")
