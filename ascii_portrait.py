"""
Draws my profile photo with characters.

    photo ──► person mask ──► edge-preserving smoothing ──► tone stretch ──► cell average ──► glyph
              (HSV + blob)     (bilateral filter)            (inside mask)    (80 × 56 grid)   (ink-measured ramp)

The eyes get extra care, because at 80 × 56 an eye is only ~5 × 2 characters:
  1. both eyes are located with OpenCV's Haar cascades;
  2. inside a soft (feathered) window around each eye and eyebrow, local detail is boosted (unsharp mask);
  3. there, a glyph is chosen by *shape* — the one whose rendered bitmap best matches the cell's pixels —
     from a small set of round and lid-like glyphs ( o O 0 @ ~ = _ - ^ ), instead of by brightness alone.

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
EYE_GLYPHS = ".,'`-_=~^*oO0@"   # round + lid-like shapes, used only around the eyes
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


def glyph_bitmaps(cell_w, cell_h, font_path=FONT, scale=4):
    """Each eye glyph rendered at the size of one grid cell, ink scaled so the densest ramp glyph = 1."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, int(round(cell_w / 0.602 * scale)))
    size = (int(round(cell_w * scale)), int(round(cell_h * scale)))
    maps = {}
    for ch in dict.fromkeys(EYE_GLYPHS + CANDIDATES.strip()):
        im = Image.new("L", size, 0)
        ImageDraw.Draw(im).text((0, 0), ch, font=font, fill=255)
        maps[ch] = cv2.resize(np.asarray(im, np.float32) / 255.0, (int(cell_w), int(cell_h)),
                              interpolation=cv2.INTER_AREA)
    top = max(m.mean() for m in maps.values())
    return {ch: maps[ch] / top for ch in EYE_GLYPHS}


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


def find_eyes(gray):
    """Up to two eye boxes (x, y, w, h): the best eye-cascade hit in each half of the upper face."""
    face_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    faces = face_c.detectMultiScale(gray, 1.1, 5, minSize=(gray.shape[1] // 4,) * 2)
    if len(faces) == 0:
        return []
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    hits = eye_c.detectMultiScale(gray[fy:fy + int(fh * 0.6), fx:fx + fw], 1.05, 4, minSize=(fw // 15,) * 2)
    eyes = []
    for left in (True, False):
        side = [b for b in hits if (b[0] + b[2] / 2 < fw / 2) == left]
        if side:
            x, y, w, h = min(side, key=lambda b: abs(b[2] - fw * 0.2))
            eyes.append((fx + x, fy + y, w, h))
    return eyes


def eye_window(shape, eyes):
    """Soft 0..1 weight around each eye and its eyebrow — feathered so the treatment leaves no seams."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    w = np.zeros(shape, np.float32)
    for x, y, bw, bh in eyes:
        cx, cy, rx, ry = x + bw / 2, y + bh * 0.42, bw * 0.85, bh * 0.75
        w = np.maximum(w, np.clip(1.4 - np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2), 0, 1))
    return cv2.GaussianBlur(w, (0, 0), 6)


def portrait(img_path, cols=80, rows=56, work_px=1000, eye_detail=2.0):
    """Everything the line renderer needs: cell tones, silhouette coverage, eye window and full-res tones."""
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

    cell_h, cell_w = gray.shape[0] / rows, gray.shape[1] / cols
    window = eye_window(gray.shape, find_eyes(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    if eye_detail:                                           # unsharp mask, only inside the eye window
        blur = cv2.GaussianBlur(gray, (0, 0), max(cell_h, cell_w) * 0.9)
        gray = np.clip(gray + eye_detail * window * (gray - blur), 0, 1)

    small = lambda a: cv2.resize(a, (cols, rows), interpolation=cv2.INTER_AREA)
    return {"tone": np.clip(small(gray), 0, 1), "cover": small(mask.astype(np.float32) / 255.0),
            "eye": small(window), "full": gray, "cell": (cell_h, cell_w)}


def to_lines(p, ramp, glyphs, invert=False, gamma=1.0, eye_thr=0.45, w_tone=8.0):
    top = len(ramp) - 1
    tone, cover, eye, (cell_h, cell_w) = p["tone"], p["cover"], p["eye"], p["cell"]
    v = np.clip(1.0 - tone if invert else tone, 0, 1) ** gamma
    vf = np.clip(1.0 - p["full"] if invert else p["full"], 0, 1) ** gamma
    gh, gw = next(iter(glyphs.values())).shape
    lines = []
    for r in range(tone.shape[0]):
        row = []
        for c in range(tone.shape[1]):
            if cover[r, c] < 0.4:
                row.append(" ")
            elif eye[r, c] > eye_thr:                        # eye area: best shape match, tone kept close
                patch = vf[int(r * cell_h):int(r * cell_h) + gh, int(c * cell_w):int(c * cell_w) + gw]
                row.append(min(glyphs, key=lambda ch: (
                    ((patch - glyphs[ch][:patch.shape[0], :patch.shape[1]]) ** 2).mean()
                    + w_tone * (patch.mean() - glyphs[ch].mean()) ** 2, ch)))
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
    p = portrait(a.image, a.cols, a.rows)
    glyphs = glyph_bitmaps(p["cell"][1], p["cell"][0])
    # gamma > 1 on the light card keeps the dark jacket from turning into a solid block of ink
    for name, inv, gamma in (("ascii_dark.txt", False, 1.0), ("ascii_light.txt", True, 1.35)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write("\n".join(to_lines(p, ramp, glyphs, invert=inv, gamma=gamma)) + "\n")
    if a.preview:
        preview(to_lines(p, ramp, glyphs), a.preview)
    print(f"ramp {ramp!r} · {a.cols}×{a.rows} · eyes found: {int((p['eye'] > 0.45).sum())} cells "
          f"· wrote ascii_dark.txt, ascii_light.txt")
