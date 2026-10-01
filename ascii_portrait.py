"""
Draws my profile photo with characters.

    photo ──► person mask ──► edge-preserving smoothing ──► tone stretch ──► cell average ──► glyph
              (HSV + blob)     (bilateral filter)            (inside mask)    (80 × 56 grid)   (ink-measured ramp)

Two versions are written, so the portrait is a *positive* image in both GitHub themes:
  ascii_dark.txt   bright pixels → dense glyphs  (light text on a dark card)
  ascii_light.txt  dark pixels   → dense glyphs  (dark text on a light card)

Usage:  python ascii_portrait.py [--image assets/portrait.jpg] [--cols 80] [--rows 56] [--preview out.png]
"""
import argparse
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"     # only used to measure glyph ink / preview
CANDIDATES = " .`'^,:;-~_=+<>!|il1/\\rcvxzunsoeajtfy?7I*JLYTZ%#&$@WMBQDNHRKOGU80"
CELL_ASPECT = 0.602 / 1.15        # glyph advance ÷ line height used in the SVG card


def build_ramp(levels=22, font_path=FONT):
    """Render every candidate glyph, measure how much ink it puts on screen, keep `levels` evenly spaced ones."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, 48)
    ink = []
    for ch in dict.fromkeys(CANDIDATES):
        im = Image.new("L", (40, 64), 0)
        ImageDraw.Draw(im).text((4, 2), ch, font=font, fill=255)
        ink.append((np.asarray(im, np.float32).mean() / 255.0, ch))
    ink.sort()
    ramp = []
    for t in np.linspace(ink[0][0], ink[-1][0], levels):
        ch = min(ink, key=lambda x: abs(x[0] - t))[1]
        if ch not in ramp:
            ramp.append(ch)
    return "".join(ramp)


def person_mask(bgr):
    """The photo was taken in front of a saturated blue wall: everything that is not wall and touches the centre."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    wall = (h >= 95) & (h <= 135) & (s >= 60) & (v >= 35)
    mask = (~wall).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = labels[mask.shape[0] // 2, mask.shape[1] // 2]
    if keep == 0:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == keep, 255, 0).astype(np.uint8)
    filled = mask.copy()                                    # fill holes (eyes, mouth)
    cv2.floodFill(filled, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(filled)


def portrait(img_path, cols=80, rows=56, work_px=1000):
    """Returns (tone grid 0..1, coverage grid 0..1) of shape rows × cols."""
    bgr = cv2.imread(img_path)
    bgr = cv2.resize(bgr, (work_px, int(work_px * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    H, W = bgr.shape[:2]
    cw = min(W, int(H * cols * CELL_ASPECT / rows))          # crop to the grid's aspect ratio
    x0 = (W - cw) // 2
    bgr = bgr[:, x0:x0 + cw]
    mask = person_mask(bgr)
    smooth = cv2.bilateralFilter(bgr, 9, 60, 9)              # keeps eyes/brows sharp, removes skin & snow noise
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lo, hi = np.percentile(gray[mask > 0], [3, 99.5])
    gray = np.clip((gray - lo) / (hi - lo + 1e-6), 0, 1)
    tone = np.clip(cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA), 0, 1)
    cover = cv2.resize(mask.astype(np.float32) / 255.0, (cols, rows), interpolation=cv2.INTER_AREA)
    return tone, cover


def to_lines(tone, cover, ramp, invert=False, gamma=1.0):
    top = len(ramp) - 1
    v = np.clip(1.0 - tone if invert else tone, 0, 1) ** gamma
    lines = []
    for r in range(tone.shape[0]):
        row = []
        for c in range(tone.shape[1]):
            if cover[r, c] < 0.4:
                row.append(" ")
            else:                                            # index ≥ 1: never a blank inside the silhouette
                row.append(ramp[int(np.clip(round(1 + v[r, c] * (top - 1)), 1, top))])
        lines.append("".join(row).rstrip())
    return lines


def preview(lines, out_png, dark=True, size=12):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, size)
    lh = size * 1.15
    bg, fg = ((22, 27, 34), (201, 209, 217)) if dark else ((246, 248, 250), (36, 41, 47))
    im = Image.new("RGB", (int(max(map(len, lines)) * size * 0.602) + 20, int(len(lines) * lh) + 20), bg)
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.text((10, 10 + i * lh), ln, font=font, fill=fg)
    im.save(out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(HERE, "assets", "portrait.jpg"))
    ap.add_argument("--cols", type=int, default=80)
    ap.add_argument("--rows", type=int, default=56)
    ap.add_argument("--preview", help="also save a PNG preview of the dark version")
    a = ap.parse_args()

    ramp = build_ramp()
    tone, cover = portrait(a.image, a.cols, a.rows)
    # gamma > 1 on the light card keeps the dark jacket from turning into a solid block of ink
    for name, inv, gamma in (("ascii_dark.txt", False, 1.0), ("ascii_light.txt", True, 1.35)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write("\n".join(to_lines(tone, cover, ramp, invert=inv, gamma=gamma)) + "\n")
    if a.preview:
        preview(to_lines(tone, cover, ramp), a.preview)
    print(f"ramp {ramp!r} · {a.cols}×{a.rows} · wrote ascii_dark.txt, ascii_light.txt")
