"""Derive every logo asset from the single supplied lockup.

The brand file arrives as one PNG: the "iL" mark plus the "ImplicitLab"
wordmark, on transparency. Everything the site needs — a dark-background
variant, the mark on its own, favicons, a social card — is generated from it
here rather than hand-edited, so there is one source of truth and regenerating
after a brand tweak is a single command.

The only interesting step is the dark-background variant. The wordmark is drawn
in two materials: "Implicit" in near-black neutral ink, and "Lab" in a
blue-to-purple gradient. On a dark UI the neutral half disappears while the
gradient half is fine, so the neutral pixels need lifting to a light colour and
the coloured ones must be left alone.

Separating them by *lightness* would fail — the deep end of the purple gradient
is as dark as the ink. They are separated by **chroma** instead
(``max(r,g,b) - min(r,g,b)``), which is near zero for neutral ink and large for
any of the brand colours, at every point along the gradient and through the
antialiased edges. That keeps the letterform edges clean instead of leaving a
pale halo around the gradient glyphs.

Usage:  ./.venv/bin/python tools/make_logo_assets.py <source.png>
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "assets"

#: Pixels with less chroma than this are treated as neutral ink and recoloured
#: for dark backgrounds. Measured values: wordmark ink ~24, brand blue ~251,
#: brand purple ~188 — so the threshold sits in a wide empty gap.
NEUTRAL_CHROMA_MAX = 45

#: The light the neutral ink becomes on a dark ground. Matches --ink in app.css.
INK_ON_DARK = (242, 243, 247)

#: Social-card ground. Matches the top of the masthead gradient.
CARD_BG = (11, 13, 18)

PAD = 6


def trim(im: Image.Image, pad: int = PAD) -> Image.Image:
    """Crop to the visible pixels, then re-add a little breathing room."""
    box = im.getchannel("A").getbbox()
    if box is None:
        return im
    im = im.crop(box)
    out = Image.new("RGBA", (im.width + pad * 2, im.height + pad * 2), (0, 0, 0, 0))
    out.paste(im, (pad, pad))
    return out


def column_is_empty(im: Image.Image, x: int, threshold: int = 8) -> bool:
    alpha = im.getchannel("A")
    return max(alpha.getpixel((x, y)) for y in range(im.height)) < threshold


def split_mark_and_wordmark(im: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Find the gutter between the mark and the wordmark and cut there.

    The widest run of fully transparent columns in the lockup is the gutter by
    construction — every gap inside a word is narrower than the space the mark
    is given.
    """
    empty = [x for x in range(im.width) if column_is_empty(im, x)]
    runs: list[list[int]] = []
    for x in empty:
        if runs and x == runs[-1][-1] + 1:
            runs[-1].append(x)
        else:
            runs.append([x])
    interior = [r for r in runs if r[0] > 0 and r[-1] < im.width - 1]
    if not interior:
        return im, im
    gutter = max(interior, key=len)
    cut = (gutter[0] + gutter[-1]) // 2
    return trim(im.crop((0, 0, cut, im.height))), trim(im.crop((cut, 0, im.width, im.height)))


def recolour_neutrals(im: Image.Image, target: tuple[int, int, int]) -> Image.Image:
    """Lift neutral ink to ``target``, leaving every chromatic pixel untouched."""
    src = im.load()
    out = Image.new("RGBA", im.size)
    dst = out.load()
    tr, tg, tb = target
    for y in range(im.height):
        for x in range(im.width):
            r, g, b, a = src[x, y]
            if a == 0:
                dst[x, y] = (0, 0, 0, 0)
                continue
            chroma = max(r, g, b) - min(r, g, b)
            dst[x, y] = (tr, tg, tb, a) if chroma < NEUTRAL_CHROMA_MAX else (r, g, b, a)
    return out


def square(im: Image.Image, size: int, margin: float = 0.10) -> Image.Image:
    """Fit ``im`` centred inside a transparent square canvas."""
    inner = int(size * (1 - margin * 2))
    scale = min(inner / im.width, inner / im.height)
    resized = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                        Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2), resized)
    return canvas


def on_ground(im: Image.Image, size: int, bg: tuple[int, int, int] = CARD_BG,
              radius: float = 0.22) -> Image.Image:
    """Same as :func:`square` but on a rounded opaque tile, for tab icons."""
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size * 4 - 1, size * 4 - 1), radius=int(size * 4 * radius), fill=255
    )
    mask = mask.resize((size, size), Image.LANCZOS)
    ground = Image.new("RGBA", (size, size), bg + (255,))
    tile.paste(ground, (0, 0), mask)
    glyph = square(im, size, margin=0.16)
    tile.alpha_composite(glyph)
    return tile


def social_card(lockup_dark: Image.Image, width: int = 1200, height: int = 630) -> Image.Image:
    """A dark card with the lockup centred and a soft brand glow behind it."""
    card = Image.new("RGBA", (width, height), CARD_BG + (255,))

    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    cx, cy = width // 2, int(height * 0.46)
    for i, (rad, alpha, col) in enumerate([
        (520, 46, (124, 92, 255)),
        (360, 40, (56, 132, 255)),
        (220, 34, (140, 110, 255)),
    ]):
        d.ellipse((cx - rad, cy - rad // 2, cx + rad, cy + rad // 2), fill=col + (alpha,))
    glow = glow.filter(ImageFilter.GaussianBlur(110))
    card.alpha_composite(glow)

    target_w = int(width * 0.70)
    scale = target_w / lockup_dark.width
    logo = lockup_dark.resize(
        (target_w, max(1, round(lockup_dark.height * scale))), Image.LANCZOS
    )
    card.alpha_composite(logo, ((width - logo.width) // 2, int(height * 0.34) - logo.height // 2))

    rule_y = int(height * 0.60)
    d2 = ImageDraw.Draw(card)
    d2.rounded_rectangle(
        (cx - 120, rule_y, cx + 120, rule_y + 3), radius=2, fill=(124, 92, 255, 190)
    )

    # The caption is typeset only if a suitable font is actually on this
    # machine. Font availability varies, and a card with a clean rule and no
    # caption is a better failure than one with a fallback bitmap face on it.
    for size, dy, colour, text in [
        (44, 0.70, (233, 235, 242), "Measuring what people don't say."),
        (25, 0.80, (150, 156, 172),
         "Implicit association testing  ·  millisecond reaction-time capture"),
    ]:
        font = _load_font(size)
        if font is None:
            continue
        box = d2.textbbox((0, 0), text, font=font)
        d2.text(
            (cx - (box[2] - box[0]) // 2, int(height * dy) - (box[3] - box[1]) // 2),
            text, font=font, fill=colour + (255,),
        )
    return card


#: Preferred first; each is tried in turn and the first that loads wins.
FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src_path = Path(sys.argv[1])
    if not src_path.exists():
        print(f"source not found: {src_path}")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    lockup = trim(Image.open(src_path).convert("RGBA"))
    mark, wordmark = split_mark_and_wordmark(lockup)

    lockup_dark = recolour_neutrals(lockup, INK_ON_DARK)

    written: list[tuple[str, Image.Image]] = [
        # The lockup exactly as supplied, for light backgrounds and print.
        ("logo-light.png", lockup),
        # The header/masthead variant.
        ("logo-dark.png", lockup_dark),
        # The mark alone. Already fully chromatic, so it needs no variant.
        ("mark.png", mark),
        ("wordmark-dark.png", recolour_neutrals(wordmark, INK_ON_DARK)),
        # Browser tab: transparent, so the mark sits on whatever colour the
        # browser paints its tab strip. A dark tile would disappear against a
        # dark theme, which is exactly where this app's users will be.
        ("favicon-32.png", square(mark, 32, margin=0.03)),
        ("favicon-64.png", square(mark, 64, margin=0.03)),
        # Home-screen icons must be opaque — iOS composites transparency onto
        # black and the result looks like a rendering bug rather than a choice.
        ("favicon-180.png", on_ground(mark, 180)),
        ("favicon-512.png", on_ground(mark, 512)),
        ("og-card.png", social_card(lockup_dark)),
    ]

    for name, img in written:
        path = OUT / name
        img.save(path, optimize=True)
        print(f"  {name:<22} {img.width:>5} x {img.height:<5}  {path.stat().st_size / 1024:>7.1f} KB")

    print(f"\nSource lockup: {lockup.width} x {lockup.height}")
    print(f"Mark:          {mark.width} x {mark.height}")
    print(f"Wordmark:      {wordmark.width} x {wordmark.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
