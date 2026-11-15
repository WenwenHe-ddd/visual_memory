#!/usr/bin/env python3
"""Draw a conceptual overview: admission, two-level memory, parallel retrieval."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visualmemory_method1_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CJK_FONT = ROOT / ".vendor/deepseek_vl2_source/deepseek_vl2/serve/assets/simsun.ttc"
INK = "#243247"
MUTED = "#69798C"
BLUE = "#327BC4"
PURPLE = "#8A63B1"
BLUE_LIGHT = "#ECF4FD"
PURPLE_LIGHT = "#F4EFFA"


def main():
    font_path = Path(os.environ.get("METHOD1_CJK_FONT", str(CJK_FONT)))
    if not font_path.is_file():
        raise FileNotFoundError("Set METHOD1_CJK_FONT to a local Chinese .ttf/.ttc font.")
    font_manager.fontManager.addfont(str(font_path))
    cjk = font_manager.FontProperties(fname=str(font_path))
    latin = font_manager.FontProperties(family="DejaVu Sans")
    # Embed vector glyphs so Chinese labels also render without a system CJK font.
    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 18), ylim=(0, 9))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def text(x, y, value, size=15, color=INK, ha="center", english=False, **kw):
        return ax.text(
            x, y, value, fontsize=size, color=color, ha=ha, va="center",
            fontproperties=latin if english else cjk, zorder=6, **kw)

    def card(x, y, w, h, fill="white", edge="none", radius=.12, lw=1.2, z=2):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=f"round,pad=0.015,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))

    def arrow(start, end, color=MUTED, lw=1.7, size=14, style="-|>"):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=size,
            linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=3))

    def line(points, color, lw=1.35, z=3):
        ax.plot([p[0] for p in points], [p[1] for p in points],
                color=color, linewidth=lw, solid_capstyle="round", zorder=z)

    def leaf(x, y, color, w=.59, h=.70):
        card(x, y, w, h, edge=color, radius=.055, lw=1.0, z=4)
        for yy, width in [(y + .14, .32), (y, .32), (y - .14, .23)]:
            line([(x - .16, yy), (x - .16 + width, yy)], color, 1.25, z=5)

    def group(x, y, name, color, light):
        # A circle encodes an aggregate vector, never a stored representative photo.
        card(x, y, 4.08, 2.24, light, radius=.18)
        ax.add_patch(Circle((x, y + .46), .43, color=color, zorder=4))
        text(x, y + .46, name, 16, "white", english=True, weight="bold")
        junction_y = y - .14
        line([(x, y + .02), (x, junction_y)], color)
        line([(x - 1.04, junction_y), (x + 1.04, junction_y)], color)
        for lx in (x - 1.04, x, x + 1.04):
            line([(lx, junction_y), (lx, y - .35)], color)
            leaf(lx, y - .69, color)

    def photo(x, y):
        # Draw an incoming image card as a simple room/sofa illustration.
        card(x + .13, y + .11, 2.63, 1.94, "#E5EAF0", radius=.10)
        card(x, y, 2.63, 1.94, "white", "#B8C6D5", radius=.10)
        ax.add_patch(Rectangle((x - 1.17, y - .60), 2.34, 1.40,
                               facecolor="#F6F0E6", edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((x - 1.17, y - .60), 2.34, .37,
                               facecolor="#E7DECF", edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((x - .95, y + .13), .50, .51,
                               facecolor="#C9E4EB", edgecolor="white", linewidth=2, zorder=4))
        line([(x - .70, y + .15), (x - .70, y + .63)], "white", 1.8, z=5)
        card(x + .22, y + .08, 1.30, .50, "#7395BB", radius=.09, z=4)
        card(x + .22, y - .21, 1.52, .42, "#4875A7", radius=.09, z=5)
        for xx in (x - .48, x + .92):
            card(xx, y - .08, .23, .48, "#6088B5", radius=.06, z=5)
        for xx in (x - .30, x + .74):
            line([(xx, y - .43), (xx, y - .52)], "#5A6773", 2.2, z=5)
        text(x, y - .79, "新到达的图片", 14, MUTED)

    # The headline explains the idea; technical details stay in the README.
    text(.45, 8.53, "METHOD 1", 12, PURPLE, ha="left", english=True, weight="bold")
    text(2.05, 8.53, "描述分组，视觉引导，文本检索", 25, ha="left")

    # Three quiet columns, with only two inter-stage arrows.
    for x, w in [(2.25, 3.80), (7.15, 4.55), (14.17, 6.95)]:
        card(x, 4.48, w, 6.94, "#F8FAFD", radius=.20)
    text(2.25, 7.52, "① 离线建库", 21)
    text(7.15, 7.52, "② 两层记忆", 21)
    text(14.17, 7.52, "③ 在线检索", 21)

    # 1. Admission: illustration plus caption card instead of encoder boxes.
    text(2.25, 6.92, "新图逐条加入", 15, MUTED)
    photo(2.25, 5.66)
    arrow((2.25, 4.64), (2.25, 4.32), BLUE)
    card(2.25, 3.73, 2.92, 1.11, "white", "#CFDAE8")
    text(2.25, 3.98, "完整 caption", 17, BLUE)
    for y, w in [(3.68, 2.22), (3.46, 1.64)]:
        line([(1.14, y), (1.14 + w, y)], "#A6B8CC", 2.0, z=5)
    text(2.25, 2.60, "Caption 决定如何分组", 17)
    text(2.25, 2.12, "图像表征用于更新 master", 15, MUTED)
    text(2.25, 1.52, "同时保留对话与时间", 14, MUTED)
    arrow((4.25, 4.48), (4.78, 4.48), "#7F8EA1", 2.0, 18)

    # 2. A literal two-level tree per cluster.
    text(7.15, 6.92, "Master 保存完整视觉表征", 16, MUTED)
    group(7.15, 5.44, "M1", BLUE, BLUE_LIGHT)
    group(7.15, 2.92, "M2", PURPLE, PURPLE_LIGHT)
    text(7.15, 1.47, "叶子：对话 + caption + 时间 + 2D 坐标", 13.5, MUTED)
    arrow((9.50, 4.48), (10.60, 4.48), "#7F8EA1", 2.0, 18)

    # 3. Two visible parallel paths, matching the memory-cluster colours.
    card(14.17, 6.45, 6.08, .99, "white", "#CFDAE8")
    text(14.17, 6.65, "问题 Q + 候选陈述 A / B / C / D", 17)
    text(14.17, 6.24, "候选陈述预先准备，每个 master 各选一条", 13, MUTED)
    for x, color in [(12.57, BLUE), (15.77, PURPLE)]:
        arrow((x, 5.91), (x, 5.55), color)

    for x, name, choice, color, light in [
        (12.57, "M1", "B", BLUE, BLUE_LIGHT),
        (15.77, "M2", "C", PURPLE, PURPLE_LIGHT),
    ]:
        card(x, 4.15, 2.82, 2.77, light, radius=.17)
        text(x, 5.19, f"{name} 视觉匹配 → 选 {choice}", 15.5, color)
        card(x, 4.57, 2.21, .65, "white", color, radius=.10)
        text(x, 4.57, f"Q + 候选 {choice}", 18, color)
        arrow((x, 4.19), (x, 3.84), color)
        for lx in (x - .69, x, x + .69):
            leaf(lx, 3.44, color, w=.51, h=.65)
        text(x, 2.97, "文本匹配本簇叶子", 14, color)

    # All groups participate; their scores merge into one ranked context.
    line([(12.57, 2.72), (12.57, 2.45), (14.17, 2.45)], BLUE, 1.7)
    line([(15.77, 2.72), (15.77, 2.45), (14.17, 2.45)], PURPLE, 1.7)
    arrow((14.17, 2.45), (14.17, 2.13), MUTED)
    card(14.17, 1.69, 6.08, .78, "white", "#CFDAE8", radius=.13)
    # Small stacked documents make Top-5 a tangible retrieved-memory bundle.
    for dx, dy in [(-.10, .08), (0, 0), (.10, -.08)]:
        card(11.75 + dx, 1.69 + dy, .36, .42, "white", BLUE, radius=.025, lw=.85, z=4)
    text(13.15, 1.69, "全局 Top-5", 18, BLUE)
    arrow((14.35, 1.69), (15.02, 1.69), MUTED)
    text(16.00, 1.69, "生成回答", 18)

    text(9, .48,
         "图中仅示意两个簇；实际所有 master 均参与。候选陈述只扩展检索 query，Top-5 叶子文本用于回答。",
         13, MUTED)

    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            OUT / f"framework.{suffix}", dpi=180, facecolor="white",
            metadata={"Creator": "VisualMemory Method 1 conceptual overview"}
            if suffix == "pdf" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
