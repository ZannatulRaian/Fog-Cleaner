"""
Urban Fog Cleaner — Dehazing Console
=====================================
Streamlit console around the image-processing pipeline in `urban.py`:
upload a hazy photo, tune the filters, download the result.

Folder layout expected next to this file:
    dehaze_app.py
    assets/cars/*.png
    assets/clouds/*.png
    .streamlit/config.toml

Run with:
    streamlit run dehaze_app.py
"""

import base64
import glob
import io
import math
import random
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import ndimage
from scipy.fft import fft2, fftshift, ifft2, ifftshift

# Optional extra for iPhone HEIC/HEIF photos (it also gives an older Pillow
# AVIF support):   pip install pillow-heif
# The app runs fine without it; such files then get a clear message instead.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

APP_DIR = Path(__file__).parent
MAX_DIM = 1200  # images larger than this (on the long edge) are downscaled


# ============================================================
# CORE PIPELINE
# ============================================================
# Every stage below returns its input untouched when its slider is at 0, so
# Reset really does switch the three filters off. The Dark Channel Prior
# fog-removal step (step 1) is the app's base and always runs.
#
# Colour handling: the smoothing / sharpening / stretch / frequency stages work
# on LUMINANCE only (the Y of YCrCb). Treating R, G and B separately shifts hues
# and paints coloured noise, which is where the purple/blue blotches came from.

FREQ_CUTOFF = 30  # frequency-domain cutoff radius, in frequency-plane pixels


def hist_clip_percent(strength):
    return (strength / 100.0) * 2.0          # 0 .. 2 % clipped from each end


def freq_gain(strength):
    return 1.0 + (strength / 100.0) * 1.5    # gain applied to fine detail


def smoothing_tolerance(strength):
    return 10.0 + 0.6 * strength             # how big a colour step still counts as an edge


def edge_gain(strength):
    return (strength / 100.0) * 2.0          # unsharp-mask amount


def _to_ycc(img):
    ycc = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2YCrCb)
    return ycc, ycc[:, :, 0] * 255.0


def _from_ycc(ycc, y):
    out = ycc.copy()
    out[:, :, 0] = np.clip(y, 0.0, 255.0) / 255.0
    rgb = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def histogram_stretch(img, strength=55, return_range=False):
    """Linear contrast stretch of the luminance, clipping `percent`% at each end."""
    percent = hist_clip_percent(strength)
    ycc, y = _to_ycc(img)
    lo, hi = np.percentile(y, [percent, 100 - percent])
    if strength <= 0 or hi - lo < 1.0:
        out = img
    else:
        out = _from_ycc(ycc, (y - lo) * 255.0 / (hi - lo))
    return (out, (float(lo), float(hi))) if return_range else out


def adaptive_smoothing(img, strength=50):
    """Edge-preserving smoothing: flat areas (and the noise the fog removal
    amplifies there) are smoothed, edges are kept. The strength sets how big a
    colour difference still counts as an edge."""
    if strength <= 0:
        return img
    sigma_color = smoothing_tolerance(strength)
    sigma_space = 3.0 + 0.05 * strength
    return cv2.bilateralFilter(img, 7, sigma_color, sigma_space)


def adaptive_sharpening(img, strength=50):
    """Unsharp mask on luminance, with the added detail capped so strong edges
    don't overshoot into white halos."""
    amount = edge_gain(strength)
    if amount <= 0:
        return img
    ycc, y = _to_ycc(img)
    detail = np.clip(y - cv2.GaussianBlur(y, (0, 0), 1.5), -40.0, 40.0)
    return _from_ycc(ycc, y + amount * detail)


def frequency_enhancement(img, strength=55):
    """High-frequency emphasis on luminance: a smooth (Gaussian) high-pass, so
    there is no ringing from a hard cutoff, and the image is mirror-padded so
    the FFT's wrap-around doesn't leak detail across opposite borders."""
    gain = freq_gain(strength)
    if gain <= 1.0:
        return img
    ycc, y = _to_ycc(img)
    h, w = y.shape
    pad = max(0, min(48, h - 1, w - 1))
    yp = np.pad(y, pad, mode="reflect") if pad else y
    rows, cols = yp.shape
    spectrum = fftshift(fft2(yp))
    v = np.arange(rows) - rows // 2
    u = np.arange(cols) - cols // 2
    d2 = v[:, None] ** 2 + u[None, :] ** 2
    high_pass = 1.0 - np.exp(-d2 / (2.0 * FREQ_CUTOFF ** 2))
    response = 1.0 + (gain - 1.0) * high_pass      # 1.0 at DC, `gain` at fine detail
    out = np.real(ifft2(ifftshift(spectrum * response)))
    out = out[pad:pad + h, pad:pad + w]
    return _from_ycc(ycc, out)


def _guided_filter(guide, src, radius, eps):
    """Edge-aware smoothing of `src` that follows the edges of `guide`
    (He et al., 'Guided Image Filtering')."""
    k = (2 * radius + 1, 2 * radius + 1)

    def mean(x):
        return cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)

    mean_g, mean_s = mean(guide), mean(src)
    cov = mean(guide * src) - mean_g * mean_s
    var = mean(guide * guide) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return mean(a) * guide + mean(b)


def dark_channel_prior(img, omega=0.95, t0=0.15):
    """Single-image haze removal (He et al.): estimate the airlight A and a
    per-pixel transmission t, then recover J = (I - A) / t + A."""
    I = img.astype(np.float32) / 255.0
    h, w = I.shape[:2]
    size = max(3, round(max(h, w) / 80)) | 1             # odd window, ~15 px at 1200 px

    dark = ndimage.minimum_filter(I.min(axis=2), size=size)

    # Airlight: of the haziest 0.1% of pixels (brightest in the dark channel),
    # take the single brightest one — one pixel, so its colour stays consistent.
    n = max(1, int(0.001 * dark.size))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    candidates = I.reshape(-1, 3)[idx]
    A = np.maximum(candidates[np.argmax(candidates.sum(axis=1))], 0.1)

    # Transmission, each colour channel normalised by ITS OWN airlight value.
    t = 1.0 - omega * ndimage.minimum_filter((I / A).min(axis=2), size=size)

    # Refine with a guided filter so t follows the photo's real edges. The raw
    # min-filter map is blocky, which shows up as blocks and halos in the result.
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    radius = max(1, min(size * 3, (min(h, w) - 1) // 2))
    t = _guided_filter(gray, t.astype(np.float32), radius, 1e-3)
    t = np.clip(t, t0, 1.0)

    J = (I - A) / t[:, :, None] + A
    return np.clip(J * 255.0 + 0.5, 0, 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def cached_dark_channel_prior(img):
    return dark_channel_prior(img)


def dehaze_pipeline(img, hist_strength=55, freq_strength=55, smooth_strength=50, sharpen_strength=50):
    """Returns (result, info). info["hist_range"] is the luminance range the
    histogram stage actually stretched, for the slider's readout."""
    result = cached_dark_channel_prior(img)                    # STEP 1 — fog removal
    result = adaptive_smoothing(result, smooth_strength)         # STEP 2
    result = adaptive_sharpening(result, sharpen_strength)       # STEP 3
    result, hist_range = histogram_stretch(result, hist_strength, return_range=True)  # STEP 4
    result = frequency_enhancement(result, freq_strength)      # STEP 5
    return result, {"hist_range": hist_range}


# ============================================================
# IMAGE HELPERS
# ============================================================

def resize_max(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        # max(1, ...) keeps a very thin image (say 1 x 5000 px) from rounding to 0
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def to_png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _readable_formats():
    """Every format Pillow can decode here, minus EPS. Reading an EPS means
    handing the file to Ghostscript (a PostScript interpreter): pointless for
    a photo tool and not something an upload box that takes any file should do."""
    Image.init()
    return [name for name in Image.OPEN if name != "EPS"]


def load_photo(file):
    """Open any image Pillow can read and return it as an 8-bit RGB array no
    bigger than MAX_DIM on its long edge.

    The uploader accepts every file type, so this has to cope with whatever
    real-world images throw at it: WebP, GIF, BMP, TIFF, AVIF, JPEG 2000, ...
    (plus HEIC with the optional pillow-heif), the camera's rotation flag,
    transparency (flattened onto white), 16-bit and float data, CMYK and
    grayscale files, and animated files (the first frame is used).
    Raises if the file isn't an image it can read — see unreadable_message().
    """
    img = Image.open(file, formats=_readable_formats())
    # Camera JPEGs can be 40+ megapixels. Let the decoder shrink them while it
    # reads (a no-op for every other format) — they are downscaled anyway.
    img.draft("RGB", (MAX_DIM, MAX_DIM))
    try:
        img = ImageOps.exif_transpose(img)  # honour the camera's rotation flag
    except Exception:
        pass  # broken EXIF block: show the picture as stored

    if img.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        # 16-bit / 32-bit / float data. Pillow's own convert() would clip it
        # at 255 and give a white image, so scale it to 8 bits ourselves.
        data = np.asarray(img, dtype=np.float32)
        top = float(data.max()) if data.size else 0.0
        if img.mode == "F" and top <= 1.0:
            gain = 255.0               # float pixels in the 0..1 range
        elif top <= 255.0:
            gain = 1.0                 # already 8-bit values in a bigger container
        else:
            # "16-bit" files often hold 10-, 12- or 14-bit data: the brightest
            # value tells which full scale to use (65535 for true 16-bit data).
            full_scale = next((2 ** bits - 1 for bits in (10, 12, 14, 16)
                               if top <= 2 ** bits - 1), top)
            gain = 255.0 / full_scale
        gray = np.clip(data * gain, 0, 255).astype(np.uint8)
        arr = np.stack([gray] * 3, axis=-1)
    elif img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        # Transparent areas would turn black in a plain RGB conversion;
        # flatten onto white (the colour of fog) instead.
        rgba = img.convert("RGBA")
        page = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        arr = np.array(Image.alpha_composite(page, rgba).convert("RGB"))
    else:
        arr = np.array(img.convert("RGB"))
    return resize_max(arr)


_RAW_EXTENSIONS = {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng",
                   ".raf", ".orf", ".rw2", ".pef", ".srw", ".x3f", ".3fr", ".erf", ".mrw", ".raw"}


def unreadable_message(name, exc):
    """A plain-language explanation for a file load_photo() could not open."""
    Image.init()  # fills Image.OPEN, which tells us if the optional codecs are installed
    ext = Path(name).suffix.lower()
    if isinstance(exc, Image.DecompressionBombError):
        return f"`{name}` is too large to open safely. Try a smaller copy of the photo."
    if ext in (".heic", ".heif", ".hif") and "HEIF" not in Image.OPEN:
        return (f"`{name}` is a HEIC/HEIF photo (the iPhone format), which needs one extra "
                "package: run `pip install pillow-heif`, restart the app and upload it again. "
                "Or save the photo as a JPG first.")
    if ext == ".avif" and "AVIF" not in Image.OPEN:
        return (f"`{name}` is an AVIF image, which needs a newer Pillow: run "
                "`pip install --upgrade pillow` (or `pip install pillow-heif`), restart the app "
                "and upload it again. Or save the image as a JPG first.")
    if ext in _RAW_EXTENSIONS:
        return (f"`{name}` is a camera RAW file, which this app can't develop. "
                "Export the photo as a JPG or TIFF first and upload that.")
    if isinstance(exc, UnidentifiedImageError):
        return (f"`{name}` doesn't look like an image this app can open. "
                "JPG, PNG, WebP, GIF, BMP and TIFF files all work.")
    reason = " ".join(str(exc).split())[:120]
    return (f"Couldn't read `{name}`" + (f" ({reason})" if reason else "") + ". The file may be "
            "damaged or incomplete — try saving it again as a JPG or PNG.")


# (label, Pillow's name for it) of the formats worth naming in the uploader's hint.
_HINT_FORMATS = [("JPG", "JPEG"), ("PNG", "PNG"), ("WebP", "WEBP"), ("GIF", "GIF"),
                 ("BMP", "BMP"), ("TIFF", "TIFF"), ("AVIF", "AVIF"), ("HEIC", "HEIF")]


def supported_formats_hint():
    """e.g. 'JPG, PNG, WebP, GIF, BMP, TIFF, AVIF and more' — only formats this
    install can really open (AVIF and HEIC depend on optional codecs)."""
    Image.init()
    return ", ".join(label for label, key in _HINT_FORMATS if key in Image.OPEN) + " and more"


# ============================================================
# PIXEL-ART HEADER (cars & clouds strip, animated left-to-right)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sprite_meta(kind):
    """Return [(base64_png, aspect_ratio), ...] for every sprite in assets/<kind>.

    The aspect ratio lets _build_sequence compute each row's *exact*
    rendered pixel width so the lane can size its own loop count safely
    (see MIN_TRACK_WIDTH below) instead of guessing.
    """
    folder = APP_DIR / "assets" / kind
    files = sorted(glob.glob(str(folder / "*.png")))
    out = []
    for f in files:
        with open(f, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        with Image.open(f) as im:
            w, h = im.size
        out.append((b64, w / h))
    return out


def _build_sequence(sprite_meta, count, height_range, gap_range, seed, top_range=None):
    """Build one randomized strip of <img> tags: random sprite, random
    height, random trailing gap (and optional vertical jitter) per image,
    so the pattern doesn't look mechanically repeated.

    Returns (html, row_width) in authored px — the caller needs the exact
    width to know how many times this row must repeat to safely outrun any
    viewport (see `_lane_html`).
    """
    rnd = random.Random(seed)
    n = len(sprite_meta)
    order = [rnd.randrange(n) for _ in range(count)]
    rnd.shuffle(order)

    parts = []
    total_w = 0.0
    for idx in order:
        b64, aspect = sprite_meta[idx]
        h = rnd.randint(*height_range)
        gap = rnd.randint(*gap_range)
        top = f"--t:{rnd.randint(*top_range)};" if top_range else ""
        # Sizes are authored in "px of a 200px-tall strip" and stored as bare
        # numbers; the iframe CSS multiplies them by --vu (1/200 of the strip's
        # real height), so the whole strip scales with the page (see render_header).
        parts.append(
            f'<img class="sprite" style="--h:{h};--g:{gap};{top}" '
            f'src="data:image/png;base64,{b64}">'
        )
        total_w += h * aspect + gap
    return "".join(parts), total_w


MIN_TRACK_WIDTH = 8000  # authored px a lane must cover before it's safe on any realistic viewport
# All sprite sizes and gaps in this file are authored for a 200px-tall strip;
# the page then scales the strip with the window (see render_header) — about
# 0.7x on a small laptop window, 2-3x on a large or zoomed-out screen — and
# every lane's real width scales with it. The strip runs full-bleed to the true
# browser edges (see inject_css), so a lane has to cover the *entire* window
# width: 8000 authored px is still ~5500 real px at the smallest scale, which
# is wider than even a 5K screen.


def _lane_html(sprite_meta, count, height_range, gap_range, seed, duration,
               top_range=None, opacity=None):
    """One infinitely-scrolling lane: a randomized row (see _build_sequence)
    repeated enough times that it's always wider than the browser window,
    then slid left by exactly one row-width on a seamless linear loop.

    Looping a *single fixed* copy-count (as an earlier version of this did)
    only stays seamless on viewports narrower than that fixed count implies;
    wider monitors run out of content before the far edge and show bare
    background. Sizing the repeat count from each row's own measured width
    instead makes every lane safe regardless of window size.
    """
    row_html, row_w = _build_sequence(sprite_meta, count, height_range, gap_range, seed, top_range)
    loops = max(3, math.ceil(MIN_TRACK_WIDTH / row_w) + 1)
    style = f"animation-duration:{duration:.2f}s;--shift:{-100 / loops:.4f}%;"
    if opacity is not None:
        style += f"opacity:{opacity};"
    layer_class = "cloud-layer" if top_range else "car-layer"
    return f'<div class="track {layer_class}" style="{style}">' + row_html * loops + "</div>"


def _embed_html(html, height):
    # st.iframe is the supported call on current Streamlit; st.components.v1.html
    # is deprecated (it prints a warning on every rerun), so it is only a fallback
    # for older installs that don't have st.iframe yet.
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        st.components.v1.html(html, height=height, scrolling=False)


def _spread_speeds(seed, lo, hi, n):
    """`n` random speeds stratified across [lo, hi] (one drawn from the
    middle of each equal sub-band) so a run of bad luck can't draw similar
    numbers and leave every lane moving at nearly the same pace — while
    still being a genuine random draw each time the seed changes, not
    fixed values."""
    rnd = random.Random(seed)
    band = (hi - lo) / n
    return [rnd.uniform(lo + (i + 0.15) * band, lo + (i + 0.85) * band) for i in range(n)]


# Column splits shared by the title and the cards. The title is rendered inside
# the same PAGE_SPLIT as the cards, so its left edge lines up with the cards'
# left edge at any window size or zoom level without hand-tuned offsets.
PAGE_SPLIT = [1, 16, 1]   # [left margin, content, right margin]
CARD_SPLIT = [5, 2]      # [Preview card, Enhancement Pipeline card]


def render_title():
    """The page title. It sits in Streamlit's own top bar (next to the Deploy
    button) rather than in a row of its own, which hands that height to the
    cards and so to the preview images — see inject_css."""
    _, title_col, _ = st.columns(PAGE_SPLIT, gap="large")
    title_col.markdown('<div class="ufc-title">Urban Fog Cleaner</div>', unsafe_allow_html=True)


def render_header():
    cars = load_sprite_meta("cars")
    clouds = load_sprite_meta("clouds")

    if not cars or not clouds:
        # Graceful fallback if the assets folder isn't next to the script:
        # just skip the animated strip (the title is rendered separately).
        return

    # Two light car lanes — deliberately sparse (fewer cars than the
    # original had), with wide random gaps for real spacing. Each lane
    # samples many more cars than are ever on screen at once (count is
    # high relative to how few are visible) purely so the repeating tile
    # cycles through a long, varied sequence before it's ever seen to
    # repeat — with only 1-2 cars per lane the same pair of colors would
    # loop right in front of you. Gap range/density is unchanged, so this
    # doesn't add visible traffic, just variety. Wide gaps mean an edge is
    # more often briefly bare, so both lanes run fast enough that any such
    # gap clears in a couple of seconds rather than sitting there.
    # The cars are ~18% smaller than in the first version (height 54-58 ->
    # 44-48); the gaps grew by the same amount, so each lane's total length
    # — and with it the number of cars on screen and how fast they move — is
    # unchanged. Cars stand on the strip's bottom edge (.car-layer's
    # padding-bottom), i.e. they span roughly 74-122 of the 200 units.
    car_speeds = _spread_speeds(seed=101, lo=7, hi=11, n=2)
    car_lane_specs = [(6, 13), (5, 31)]  # (cars in lane, seed)
    car_layers_html = "".join(
        _lane_html(cars, count=count, height_range=(44, 48), gap_range=(160, 270),
                   seed=seed, duration=duration)
        for (count, seed), duration in zip(car_lane_specs, car_speeds)
    )

    # Two cloud depth-layers (mid/far) — the biggest near layer (70-98 units
    # tall) was dropped: those clouds dominated the strip. The clouds start
    # right at the strip's top edge (the header's end line) and hang low
    # enough that their bottoms still overlap the cars (cloud bottoms land
    # around 51-101, the cars span ~74-122) so the fog reads as drifting
    # behind/around the cars. Clouds are emitted before cars in
    # the HTML, so cars paint on top of any cloud they overlap — that is what
    # makes the fog read as drifting behind/around the cars.
    cloud_layers_html = (
        _lane_html(clouds, count=9, height_range=(51, 75), gap_range=(31, 71),
                   seed=51, duration=61, top_range=(0, 24), opacity=0.80)
        + _lane_html(clouds, count=9, height_range=(37, 53), gap_range=(23, 55),
                     seed=83, duration=79, top_range=(20, 46), opacity=0.62)
    )

    html = f"""
    <div class="ufc-strip">
      {cloud_layers_html}
      {car_layers_html}
    </div>
    <style>
      * {{ box-sizing:border-box; }}
      html, body {{ margin:0; background:#f5f5f5; overflow:hidden; }}
      /* The page scales with the window (see inject_css) and sets this
         iframe's height in rem, so the strip's own height is the one thing
         that tracks that scale. Every sprite size, gap and offset below is
         authored in "px of a 200px-tall strip" (a bare number in the inline
         style) and multiplied by --vu, 1/200 of the real strip height — so
         cars and clouds grow and shrink in step with the cards underneath
         instead of staying a fixed size on a very large or zoomed-out
         screen. */
      :root {{ --vu: 0.5vh; }}
      .ufc-strip {{
        position:relative; height:100vh; overflow:hidden;
        border-top:1px solid rgba(0,0,0,.07);
      }}
      /* Each lane is N copies of one randomized row, slid left by exactly
         one row-width (--shift, a per-lane % set inline by _lane_html) on
         a seamless linear loop — copy N+1 always sits exactly where copy N
         started, so the wrap-around is invisible. --shift is computed from
         each row's own measured width, not a fixed guess, which is what
         keeps this gap-free on any window size (see MIN_TRACK_WIDTH). */
      .track {{ position:absolute; left:0; top:0; height:100%; display:flex;
        animation-name: scrollLTR; animation-timing-function: linear;
        animation-iteration-count: infinite; will-change: transform; }}
      .cloud-layer {{ align-items:flex-start; padding-top:0; }}
      .car-layer {{ align-items:flex-end; padding-bottom:calc(78 * var(--vu)); }}
      img.sprite {{
        image-rendering:pixelated; display:block; flex-shrink:0;
        height:calc(var(--h) * var(--vu));
        margin-right:calc(var(--g) * var(--vu));
        margin-top:calc(var(--t, 0) * var(--vu));
      }}
      @keyframes scrollLTR {{
        from {{ transform:translateX(var(--shift)); }}
        to   {{ transform:translateX(0%); }}
      }}
    </style>
    """
    _embed_html(html, height=200)


# ============================================================
# PAGE-WIDE CSS (fonts + card typography, no widget-overlay hacks)
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        /* ------------------------------------------------------------
           ONE SCALE FOR THE WHOLE PAGE
           Streamlit sizes almost everything (padding, buttons, sliders,
           toolbar, gaps) in rem, and everything below is written in rem
           too, so setting the root font size once scales the entire UI
           together. It follows the window — whichever of width/height
           is the tighter fit — so the page looks the same whether it is
           on a laptop, a big monitor, or a browser that is zoomed out,
           instead of being a fixed-pixel layout that turns tiny on big
           screens. ~17.6px at ~1820x940, floored so text stays legible on
           small windows and capped on huge ones.
           ------------------------------------------------------------ */
        html {
            font-size: clamp(11px, min(0.968vw, 1.84vh), 48px) !important;

            /* Height of the two cards: whatever is left of the window under
               the top bar, the strip and the gap below it (3.75 + 7.5 + 1.2
               = 12.45rem), minus the page's bottom padding (1.2rem) and a
               0.15rem allowance for rounding — so the page ends just inside
               the bottom of the window and never scrolls — but no more than
               40rem on a very tall window. The preview images are sized from
               this (see the preview rules below). */
            --ufc-card-h: min(calc(100vh - 13.8rem), 40rem);
        }

        [data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]) {
            font-family:'IBM Plex Sans', sans-serif;
        }

        /* Page frame. The title lives in Streamlit's own 3.75rem top bar, so
           the content starts right at the top (no top padding) and the title
           block is exactly as tall as the bar. The bottom padding is kept
           small so the whole app fits one screen. The side padding is
           Streamlit's own 5rem; max-width only matters on very wide windows,
           where it stops the cards stretching into long thin bars. */
        [data-testid="stMain"] { overflow-x: hidden; }
        [data-testid="stMainBlockContainer"] {
            padding: 0 2rem 1.2rem 2rem;
            max-width: 112rem;
            margin-left: auto;
            margin-right: auto;
        }
        /* Vertical spacing between the title, strip and cards is set
           explicitly below instead of by Streamlit's default 1rem gap
           between every element. */
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0; }

        /* The blanket IBM Plex rule above has specificity (0,2,0), which beats
           a lone class selector, so every rule that wants a different face
           (display headings, monospace numbers) is written with the same
           prefix, further down, so it wins on order. */
        [data-testid="stAppViewContainer"] .ufc-title,
        [data-testid="stAppViewContainer"] .pipeline-heading,
        [data-testid="stAppViewContainer"] .card-heading {
            font-family:'Big Shoulders Display', sans-serif;
        }
        [data-testid="stAppViewContainer"] .stage-num,
        [data-testid="stAppViewContainer"] .stage-value {
            font-family:'IBM Plex Mono', monospace;
        }

        /* ---------------- title ---------------- */
        /* 2.3rem of text + .725rem above and below = 3.75rem, the height of
           Streamlit's top bar, so the title is centred in that bar on the
           same line as the Deploy button. The bar is opaque and paints over
           the page, so the title's container is lifted just above it. */
        .ufc-title {
            font-weight:800;
            font-size:2.3rem; line-height:1; text-transform:uppercase;
            letter-spacing:.03em; color:#111; margin:.725rem 0;
        }
        [data-testid="stMarkdownContainer"]:has(> .ufc-title) {
            position: relative;
            z-index: 999991;
        }
        /* st.markdown wraps text in a container with a -1rem bottom margin
           (it expects the paragraph's own bottom margin to cancel it). The
           title and the readouts below have no such margin, so cancel it. */
        [data-testid="stMarkdownContainer"]:has(> .ufc-title),
        .st-key-pipeline_card [data-testid="stMarkdownContainer"] {
            margin-bottom: 0 !important;
        }

        /* ---------------- animated strip ----------------
           Break the header iframe out of Streamlit's centered content
           column so the cars/clouds strip runs edge-to-edge of the actual
           browser window. !important is needed because Streamlit sets an
           explicit width/height on these elements. Its height is in rem so
           it scales with everything else (the sprites inside scale with
           this height, see render_header). */
        div[data-testid="stElementContainer"]:has(iframe[data-testid="stIFrame"]) {
            width: 100vw !important;
            max-width: 100vw !important;
            /* Streamlit gives this wrapper `flex: 0 0 200px`, and flex-basis
               wins over height — so without resetting it the wrapper could
               never be shorter than 200px on a small window. */
            flex: 0 0 auto !important;
            height: 7.5rem !important;
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw !important;
            margin-right: -50vw !important;
            margin-bottom: 1.2rem;
        }
        iframe[data-testid="stIFrame"] {
            height: 7.5rem !important;
            display: block;
        }

        /* ---------------- cards ---------------- */
        /* Both cards always have the same, window-filling height (see
           --ufc-card-h), whether or not a photo is loaded, so nothing jumps
           when one is. position:relative anchors the "Processing…" pill. */
        .st-key-upload_card, .st-key-pipeline_card {
            padding: 1.5rem 1.75rem;
            min-height: var(--ufc-card-h);
            position: relative;
        }
        /* Buttons scale with the text instead of staying at Streamlit's
           fixed small size. (Not the little fullscreen button that appears
           on an image when it is hovered — that one is meant to stay small.) */
        .st-key-upload_card button:not([data-testid="stBaseButton-elementToolbar"]),
        .st-key-pipeline_card button:not([data-testid="stBaseButton-elementToolbar"]) {
            font-size: 1.05rem;
            min-height: 2.8rem;
        }

        /* Equal-height cards. Streamlit already stretches both columns to
           the taller one, but the wrapper around each card doesn't grow to
           fill its column, so the shorter card just stops early. Letting
           that wrapper grow makes the card fill the column; the upload
           area then grows to fill its card, so the dropzone simply gets
           roomier instead of leaving dead space under it. */
        [data-testid="stLayoutWrapper"]:has(> .st-key-upload_card),
        [data-testid="stLayoutWrapper"]:has(> .st-key-pipeline_card) {
            flex: 1 1 auto;
        }
        .st-key-upload_card > [data-testid="stElementContainer"]:has([data-testid="stFileUploader"]) {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
        }
        .st-key-upload_card [data-testid="stFileUploader"] {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
        }

        /* ---------------- upload dropzone ----------------
           One big drop target with its contents centered: an upload icon,
           the prompt, the red Upload button and the size hint. The icon and
           prompt are drawn with ::before / ::after (the real label is
           hidden in the uploader call and kept as the dropzone's aria-label)
           and `order` slots them around the button that Streamlit renders. */
        .st-key-upload_card [data-testid="stFileUploaderDropzone"] {
            flex: 1 1 auto;
            min-height: 15rem;
            padding: 1.5rem 1.75rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: .85rem;
            border: 2px dashed rgba(0,0,0,.18);
            border-radius: .9rem;
            background: #fafafa;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzone"]::before {
            content: "";
            order: 1;
            width: 3.6rem;
            height: 3.6rem;
            opacity: .6;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 15.5V4.5'/%3E%3Cpath d='M7.5 9 12 4.5 16.5 9'/%3E%3Cpath d='M4.5 14.5v3a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-3'/%3E%3C/svg%3E") center / contain no-repeat;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzone"]::after {
            content: "Drop a foggy photo here";
            order: 2;
            font-size: 1.45rem;
            font-weight: 600;
            line-height: 1.2;
            color: #111;
            margin-bottom: .4rem;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzone"] > span { order: 3; }
        .st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"] {
            order: 4;
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"] span,
        .st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"] small {
            font-size: .92rem;
        }
        /* The uploader no longer filters by file type, so Streamlit's own hint
           shows only the size limit; this adds the formats this install opens. */
        .st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content: "__FORMAT_HINT__";
            font-size: .92rem;
            color: rgba(49, 51, 63, .6);
            margin-top: .15rem;
        }
        /* "can't open this file" message: the file name is set in code style,
           which Streamlit shrinks to 75% — keep it in step with the sentence. */
        .st-key-upload_card [data-testid="stAlert"] code {
            font-size: .92em;
        }
        /* Red, chunkier "Upload" button inside the dropzone */
        .st-key-upload_card [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
            background-color: #d32f2f;
            border-color: #d32f2f;
            color: #ffffff;
            padding: .7rem 2.1rem;
            font-size: 1.1rem;
            border-radius: .4rem;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #b71c1c;
            border-color: #b71c1c;
            color: #ffffff;
        }

        /* ---------------- preview images ----------------
           After an upload the pair is as big as the card allows, whatever
           the photo's shape. Python publishes the photo's width/height ratio
           as --pv-ar (see the app section). --pv-maxh is the tallest a
           picture can be before the card (--ufc-card-h) runs out of room:
           card height minus its padding (3rem), the heading row (2.8rem),
           the gap under that (1rem), the caption (1.5rem) and the card's
           border plus some slack (0.2rem). The block is then sized to
           exactly two pictures of that height plus the 1rem between them,
           or to the card's inner width if that is smaller. The two columns
           inside it, and the images filling them, therefore fit their
           pictures exactly: no letterboxing, no max-height clipping, and
           Streamlit's own hover-fullscreen button stays at the corner of
           the picture itself. */
        .st-key-upload_card {
            --pv-maxh: max(12rem, calc(var(--ufc-card-h) - 8.5rem));
        }
        .st-key-preview_images {
            flex: 0 1 auto !important;
            width: min(100%, calc(2 * var(--pv-maxh) * var(--pv-ar, 1.25) + 1rem)) !important;
            margin-left: auto;
            margin-right: auto;
        }
        .st-key-preview_images [data-testid="stImage"] img {
            width: 100% !important;
            height: auto;
        }
        .st-key-preview_images [data-testid="stImageCaption"] {
            margin-top: .3rem;
            padding: 0;
        }
        .st-key-preview_images [data-testid="stImageCaption"] p {
            font-size: .9rem;
            line-height: 1.3;
        }
        /* Any height the pair doesn't need (a wide, short photo) is split
           above and below it instead of leaving a blank strip at the bottom. */
        [data-testid="stLayoutWrapper"]:has(> .st-key-preview_images) {
            margin-top: auto;
            margin-bottom: auto;
        }
        /* st.spinner adds a row to the card (plus a gap) while it works,
           which would make the card — and the page — jump. Float it instead,
           in the free space of the heading row so it never covers a photo. */
        .st-key-upload_card > [data-testid="stElementContainer"]:has(> [data-testid="stSpinner"]) {
            position: absolute;
            top: 2.1rem;
            left: 10.5rem;
            z-index: 3;
            width: auto;
            padding: .3rem .9rem;
            background: #ffffff;
            border: 1px solid rgba(0,0,0,.1);
            border-radius: 999px;
            box-shadow: 0 .1rem .5rem rgba(0,0,0,.12);
        }

        /* ---------------- pipeline card ---------------- */
        /* The card is as tall as the window allows, so any height beyond what
           the controls need is shared out between the sections (the margins
           below are just the minimum between them). */
        .st-key-pipeline_card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0;
        }
        /* Each stage (label row + slider + readout) sits in its own keyed
           container so its internal spacing is exact and the space BETWEEN
           stages is clearly bigger than the space inside one — that is
           what makes each readout read as belonging to its own slider. */
        .st-key-stage_hist, .st-key-stage_freq, .st-key-stage_smooth, .st-key-stage_sharpen {
            gap: .15rem;
            margin-top: .9rem;
        }
        .st-key-pipeline_actions {
            gap: .7rem;
            margin-top: 1.2rem;
        }

        .pipeline-heading {
            text-transform:uppercase;
            font-weight:800; font-size:2rem; line-height:1.1; letter-spacing:.02em; color:#111;
            border-bottom:1px solid rgba(0,0,0,.08); padding-bottom:.8rem;
        }
        .card-heading {
            text-transform:uppercase;
            font-weight:800; font-size:2rem; line-height:1.1; letter-spacing:.02em; color:#111;
        }
        .stage-row {
            display:flex; justify-content:space-between; align-items:baseline;
            line-height:1.3;
        }
        .stage-num {
            color:#d32f2f; font-size:.85rem;
            margin-right:.55rem;
        }
        .stage-label { font-weight:600; font-size:1.08rem; color:#111; }
        .stage-value {
            color:#d32f2f; font-weight:600; font-size:1.08rem;
        }
        /* Specific enough (p + container) to beat Streamlit's own
           `.st-emotion-cache-xxxx p { font-size:inherit; margin:0 }` rule,
           which otherwise silently overrides a plain class selector. */
        [data-testid="stMarkdownContainer"] p.stage-readout {
            font-family:'IBM Plex Mono', monospace; font-size:.84rem; line-height:1.3;
            color:#777777; margin:0;
        }

        /* Sliders: a compact track with a comfortably sized thumb. The big
           (20px) padding Streamlit puts above and below the track is only
           there to hold the value bubble and the min/max ticks; neither is
           shown (the live value is in each stage's row), so it is trimmed. */
        .st-key-pipeline_card [data-testid="stSlider"] > div > div {
            padding: .75rem 0;
        }
        .st-key-pipeline_card [data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]) {
            width: 1.1rem !important;
            height: 1.1rem !important;
        }
        [data-testid="stSliderTickBar"] { display: none; }
        /* The little value bubble above the thumb is only shown while the
           slider is hovered or being dragged. At rest it would sit on top
           of the label row above it (the live value is always shown at the
           right end of that row anyway). :focus-visible is deliberately not
           used — after a mouse drag the browser keeps the hidden range input
           "focus-visible", which would leave the bubble stuck on screen. */
        [data-testid="stSliderThumbValue"] {
            opacity: 0;
            pointer-events: none;
            transition: opacity .12s ease;
            font-size: .88rem;
            font-weight: 600;
            background: #ffffff;
            border: 1px solid rgba(0,0,0,.12);
            border-radius: .35rem;
            padding: .05rem .4rem;
            box-shadow: 0 .1rem .45rem rgba(0,0,0,.14);
            z-index: 5;
        }
        [data-testid="stSlider"]:hover [data-testid="stSliderThumbValue"],
        [data-testid="stSlider"]:active [data-testid="stSliderThumbValue"] {
            opacity: 1;
        }
        </style>
        """.replace("__FORMAT_HINT__", supported_formats_hint()),
        unsafe_allow_html=True,
    )


# ============================================================
# APP
# ============================================================

st.set_page_config(page_title="Urban Fog Cleaner", page_icon="🌫️", layout="wide")
inject_css()
render_title()
render_header()

if "source_key" not in st.session_state:
    st.session_state.source_key = None
if "source_img" not in st.session_state:
    st.session_state.source_img = None

st.session_state.setdefault("hist_strength", 55)
st.session_state.setdefault("freq_strength", 55)
st.session_state.setdefault("smooth_strength", 50)
st.session_state.setdefault("sharpen_strength", 50)
st.session_state.setdefault("upload_round", 0)     # bumped to hand the user a fresh, empty uploader
st.session_state.setdefault("upload_error", None)  # why the last file couldn't be opened


def _apply_auto_enhance():
    st.session_state.hist_strength = 55
    st.session_state.freq_strength = 55
    st.session_state.smooth_strength = 50
    st.session_state.sharpen_strength = 50


def _apply_reset():
    st.session_state.hist_strength = 0
    st.session_state.freq_strength = 0
    st.session_state.smooth_strength = 0
    st.session_state.sharpen_strength = 0


has_image = st.session_state.source_img is not None
pipe_info = None  # filled in by the Preview card once a photo is loaded

# Both cards live inside a narrower, centered strip rather than stretching
# across the whole wide-mode page — full-width made them read as short,
# flattened bars instead of the more upright cards they're meant to be.
_, center_col, _ = st.columns(PAGE_SPLIT, gap="large")
left_col, right_col = center_col.columns(CARD_SPLIT, gap="medium")

# ------------------------------------------------------------
# LEFT CARD — upload / preview
# ------------------------------------------------------------
with left_col:
    with st.container(border=True, key="upload_card"):
        if not has_image:
            # No `type=` filter: every image format is accepted (WebP, GIF, BMP,
            # TIFF, ... — see load_photo). A file that isn't a readable image gets
            # an explanation under the box, and the box starts over (a new widget
            # key means a fresh, empty uploader) so the next try is one click away.
            uploaded_file = st.file_uploader(
                "Drop a foggy photo here",
                label_visibility="collapsed",
                key=f"photo_uploader_{st.session_state.upload_round}",
            )
            if uploaded_file is not None:
                try:
                    photo = load_photo(uploaded_file)
                except Exception as exc:
                    st.session_state.upload_error = unreadable_message(uploaded_file.name, exc)
                    st.session_state.upload_round += 1
                else:
                    st.session_state.upload_error = None
                    st.session_state.source_img = photo
                    st.session_state.source_key = f"{uploaded_file.name}-{uploaded_file.size}"
                st.rerun()
            if st.session_state.upload_error:
                st.error(st.session_state.upload_error)
        else:
            # The photo's shape drives the size of the preview pair (see the
            # "preview images" CSS). It travels as a CSS variable in a <style>
            # tag that rides along with the heading, so it adds no element
            # (and no gap) of its own.
            src_h, src_w = st.session_state.source_img.shape[:2]
            # Only for a sliver of a picture (taller than 5:1, e.g. a long scroll
            # screenshot): its columns would get too narrow for the captions to
            # stay on one line, and can't go below 1rem at all, which would
            # stretch the picture far past the card. So give the pair a minimum
            # width and cap the picture's height (it is then centred in its
            # column). Normal photos get no extra rule at all.
            sliver_cap = (
                '.st-key-preview_images { width: min(100%, max('
                'calc(2 * var(--pv-maxh) * var(--pv-ar) + 1rem), 9rem)) !important; }'
                '.st-key-preview_images [data-testid="stImage"] img '
                '{ max-height: var(--pv-maxh); object-fit: contain; }'
                if src_w / src_h < 0.2 else ""
            )
            top_l, top_r = st.columns([3, 1])
            with top_l:
                st.markdown(
                    f'<style>.st-key-preview_images {{ --pv-ar: {src_w / src_h:.4f}; }}{sliver_cap}</style>'
                    '<div class="card-heading">Preview</div>',
                    unsafe_allow_html=True,
                )
            with top_r:
                if st.button("Change Image", width="stretch"):
                    st.session_state.source_img = None
                    st.session_state.source_key = None
                    st.rerun()

            with st.spinner("Processing…"):
                result, pipe_info = dehaze_pipeline(
                    st.session_state.source_img,
                    hist_strength=st.session_state.hist_strength,
                    freq_strength=st.session_state.freq_strength,
                    smooth_strength=st.session_state.smooth_strength,
                    sharpen_strength=st.session_state.sharpen_strength,
                )

            with st.container(key="preview_images"):
                col_before, col_after = st.columns(2)
                with col_before:
                    st.image(st.session_state.source_img, caption="Original", width="stretch")
                with col_after:
                    st.image(result, caption="Dehazed", width="stretch")


# ------------------------------------------------------------
# RIGHT CARD — enhancement pipeline
# ------------------------------------------------------------
def _stage_row(num, label, value):
    return (
        f'<div class="stage-row"><span><span class="stage-num">{num}</span>'
        f'<span class="stage-label">{label}</span></span>'
        f'<span class="stage-value">{value}</span></div>'
    )


with right_col:
    with st.container(border=True, key="pipeline_card"):
        st.markdown('<div class="pipeline-heading">Enhancement Pipeline</div>', unsafe_allow_html=True)

        # Stage 01 — Histogram Stretch
        with st.container(key="stage_hist"):
            st.markdown(_stage_row("01", "Histogram Stretch", st.session_state.hist_strength),
                        unsafe_allow_html=True)
            hist_strength = st.slider("Histogram strength", 0, 100, key="hist_strength",
                                       label_visibility="collapsed")
            percent = hist_clip_percent(hist_strength)
            if has_image and pipe_info:
                lo, hi = pipe_info["hist_range"]   # measured on this stage's own input
                range_txt = f"{int(lo)}&ndash;{int(hi)}"
            else:
                range_txt = "n/a"
            st.markdown(
                f'<p class="stage-readout">Clip {percent:.1f}% &middot; Range {range_txt}</p>',
                unsafe_allow_html=True,
            )

        # Stage 02 — Frequency-Domain Filter
        with st.container(key="stage_freq"):
            st.markdown(_stage_row("02", "Frequency-Domain Filter", st.session_state.freq_strength),
                        unsafe_allow_html=True)
            freq_strength = st.slider("Frequency strength", 0, 100, key="freq_strength",
                                       label_visibility="collapsed")
            boost = freq_gain(freq_strength)
            cutoff_txt = f"{FREQ_CUTOFF}px" if has_image else "n/a"
            st.markdown(
                f'<p class="stage-readout">Boost &times;{boost:.2f} &middot; Cutoff {cutoff_txt}</p>',
                unsafe_allow_html=True,
            )

        # Stage 03 — Adaptive Smoothing
        with st.container(key="stage_smooth"):
            st.markdown(_stage_row("03", "Adaptive Smoothing", st.session_state.smooth_strength),
                        unsafe_allow_html=True)
            smooth_strength = st.slider("Smoothing strength", 0, 100, key="smooth_strength",
                                         label_visibility="collapsed")
            tolerance_txt = f"{smoothing_tolerance(smooth_strength):.0f}" if smooth_strength > 0 else "off"
            st.markdown(
                f'<p class="stage-readout">Edge tolerance {tolerance_txt}</p>',
                unsafe_allow_html=True,
            )

        # Stage 04 — Adaptive Sharpening
        with st.container(key="stage_sharpen"):
            st.markdown(_stage_row("04", "Adaptive Sharpening", st.session_state.sharpen_strength),
                        unsafe_allow_html=True)
            sharpen_strength = st.slider("Sharpening strength", 0, 100, key="sharpen_strength",
                                          label_visibility="collapsed")
            amount = edge_gain(sharpen_strength)
            st.markdown(
                f'<p class="stage-readout">Edge gain &times;{amount:.2f}</p>',
                unsafe_allow_html=True,
            )

        with st.container(key="pipeline_actions"):
            b1, b2 = st.columns(2)
            with b1:
                st.button("Auto-Enhance", width="stretch", on_click=_apply_auto_enhance)
            with b2:
                st.button("Reset", width="stretch", on_click=_apply_reset)

            if has_image:
                st.download_button(
                    "Download Result",
                    data=to_png_bytes(result),
                    file_name="dehazed_result.png",
                    mime="image/png",
                    type="primary",
                    width="stretch",
                )
            else:
                st.download_button(
                    "Download Result",
                    data=b"",
                    file_name="dehazed_result.png",
                    mime="image/png",
                    type="primary",
                    width="stretch",
                    disabled=True,
                )
