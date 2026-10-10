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
from PIL import Image
from scipy import ndimage
from scipy.fft import fft2, fftshift, ifft2, ifftshift

APP_DIR = Path(__file__).parent
MAX_DIM = 1200  # images larger than this (on the long edge) are downscaled


# ============================================================
# CORE PIPELINE — copied from urban.py
# ============================================================

def histogram_stretch(img, strength=55):
    percent = 0.5 + (strength / 100.0) * 9.5
    result = np.zeros_like(img)

    for i in range(3):
        channel = img[:, :, i]
        lo, hi = np.percentile(channel, [percent, 100 - percent])

        if hi > lo:
            stretched = (channel.astype(np.float32) - lo) * 255.0 / (hi - lo)
            result[:, :, i] = np.clip(stretched, 0, 255).astype(np.uint8)
        else:
            result[:, :, i] = channel

    return result


def adaptive_smoothing(img, strength=50):
    sigma = 1.0 + (strength / 100.0) * 2.0
    result = cv2.bilateralFilter(img, 9, 75, 75)
    return result


def adaptive_sharpening(img, strength=50):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    amount = 0.5 + (strength / 100.0) * 1.5

    sharpened = (
        img.astype(np.float32)
        + amount * (img.astype(np.float32) - blurred.astype(np.float32))
    )

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def frequency_enhancement(img, strength=55):
    img_float = img.astype(np.float32)
    boost = 0.5 + (strength / 100.0) * 1.5

    enhanced_channels = []
    for i in range(3):
        channel = img_float[:, :, i]

        frequency = fft2(channel)
        frequency_shifted = fftshift(frequency)

        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        high_pass = np.zeros((rows, cols), dtype=np.float32)
        high_pass[distance > 30] = 1.0

        high_frequency = frequency_shifted * high_pass
        detail = ifft2(ifftshift(high_frequency))
        detail = np.real(detail)

        detail_mean = np.mean(detail)
        detail = detail - detail_mean
        detail_std = np.std(detail)
        if detail_std > 0:
            detail = detail / detail_std

        detail = np.clip(detail, -3, 3)
        detail = detail * 10.0

        enhanced = channel + detail * boost
        enhanced_channels.append(np.clip(enhanced, 0, 255).astype(np.uint8))

    return np.stack(enhanced_channels, axis=2)


def dark_channel_prior(img, omega=0.95, t0=0.1, radius=15):
    img_float = img.astype(np.float32) / 255.0
    min_img = np.min(img_float, axis=2)
    dark = ndimage.minimum_filter(min_img, size=radius)

    flat_dark = dark.flatten()
    flat_img = img_float.reshape(-1, 3)
    number_pixels = max(1, int(0.001 * len(flat_dark)))
    top_indices = np.argsort(flat_dark)[-number_pixels:]
    A = np.max(flat_img[top_indices], axis=0)

    transmission = 1 - omega * ndimage.minimum_filter(
        min_img / max(np.min(A), 1e-6), size=radius
    )
    transmission = np.clip(transmission, t0, 1.0)

    result = np.zeros_like(img_float)
    for i in range(3):
        result[:, :, i] = (img_float[:, :, i] - A[i]) / transmission + A[i]

    return np.clip(result * 255, 0, 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def cached_dark_channel_prior(img):
    return dark_channel_prior(img)


def dehaze_pipeline(img, hist_strength=55, freq_strength=55, adaptive_strength=50):
    result = cached_dark_channel_prior(img)          # STEP 1 — Dark Channel Prior
    result = adaptive_smoothing(result, adaptive_strength)   # STEP 2
    result = adaptive_sharpening(result, adaptive_strength)  # STEP 3
    result = histogram_stretch(result, hist_strength)        # STEP 4
    result = frequency_enhancement(result, freq_strength)    # STEP 5
    return result


# ============================================================
# IMAGE HELPERS
# ============================================================

def resize_max(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def to_png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


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


def render_title():
    """The page title, rendered as a normal Streamlit element inside the same
    1/7/1 column split the cards use, so its left edge lines up with the cards
    at any window size or zoom level without any hand-tuned offsets."""
    _, title_col, _ = st.columns([1, 7, 1], gap="large")
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
    car_speeds = _spread_speeds(seed=101, lo=7, hi=11, n=2)
    car_lane_specs = [(6, 13), (5, 31)]  # (cars in lane, seed)
    car_layers_html = "".join(
        _lane_html(cars, count=count, height_range=(54, 58), gap_range=(140, 240),
                   seed=seed, duration=duration)
        for (count, seed), duration in zip(car_lane_specs, car_speeds)
    )

    # Three cloud depth-layers (near/mid/far), sized and positioned to
    # reach down into the same band the cars occupy (cars sit roughly in
    # the strip's bottom third — see .car-layer's padding-bottom below)
    # instead of staying confined to a strip of "sky" above them. Clouds
    # are still emitted before cars in the HTML, so cars paint on top of
    # any cloud they overlap — this is what makes the fog read as drifting
    # behind/around the cars rather than sitting in a separate band above
    # them. Gaps are wider than before (fewer, more spread-out clouds).
    cloud_layers_html = (
        _lane_html(clouds, count=9, height_range=(85, 120), gap_range=(15, 35),
                   seed=7, duration=44, top_range=(25, 55), opacity=0.95)
        + _lane_html(clouds, count=9, height_range=(62, 92), gap_range=(13, 28),
                     seed=51, duration=61, top_range=(45, 75), opacity=0.80)
        + _lane_html(clouds, count=9, height_range=(45, 65), gap_range=(11, 24),
                     seed=83, duration=79, top_range=(65, 95), opacity=0.62)
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
      .cloud-layer {{ align-items:flex-start; padding-top:calc(6 * var(--vu)); }}
      .car-layer {{ align-items:flex-end; padding-bottom:calc(12 * var(--vu)); }}
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
        }

        [data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]) {
            font-family:'IBM Plex Sans', sans-serif;
        }

        /* Page frame. The header toolbar is 3.75rem tall and sits on top of
           the page, so the content starts just below it; the bottom padding
           is kept small so the whole app fits one screen. The side padding is
           Streamlit's own 5rem; max-width only matters on very wide windows,
           where it stops the cards stretching into long thin bars. */
        [data-testid="stMain"] { overflow-x: hidden; }
        [data-testid="stMainBlockContainer"] {
            padding: 3.75rem 5rem 1.5rem 5rem;
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
        .ufc-title {
            font-weight:800;
            font-size:2.3rem; line-height:1; text-transform:uppercase;
            letter-spacing:.03em; color:#111; margin:.6rem 0 .75rem 0;
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
            height: 12.5rem !important;
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw !important;
            margin-right: -50vw !important;
            margin-bottom: 1.2rem;
        }
        iframe[data-testid="stIFrame"] {
            height: 12.5rem !important;
            display: block;
        }

        /* ---------------- cards ---------------- */
        .st-key-upload_card, .st-key-pipeline_card {
            padding: 1.5rem 1.75rem;
        }
        /* Buttons scale with the text instead of staying at Streamlit's
           fixed small size. */
        .st-key-upload_card button, .st-key-pipeline_card button {
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

        /* After an upload: keep the two preview images inside the card's
           height (a tall portrait photo would otherwise push the page into
           a scroll), whatever their shape. */
        .st-key-upload_card [data-testid="stImage"] img {
            max-height: 21rem;
            object-fit: contain;
        }
        .st-key-upload_card [data-testid="stImageCaption"] { font-size: .95rem; }
        /* The card is as tall as the pipeline card next to it, so a wide
           (short) photo pair would leave a blank strip along the bottom.
           Auto margins on the preview block centre it in the space under
           the heading instead. */
        [data-testid="stLayoutWrapper"]:has(> .st-key-preview_images) {
            margin-top: auto;
            margin-bottom: auto;
        }

        /* ---------------- pipeline card ---------------- */
        .st-key-pipeline_card {
            display: flex;
            flex-direction: column;
            gap: 0;
        }
        /* Each stage (label row + slider + readout) sits in its own keyed
           container so its internal spacing is exact and the space BETWEEN
           stages is clearly bigger than the space inside one — that is
           what makes each readout read as belonging to its own slider. */
        .st-key-stage_hist, .st-key-stage_freq, .st-key-stage_adapt {
            gap: .15rem;
            margin-top: 1.25rem;
        }
        .st-key-pipeline_actions {
            gap: .7rem;
            margin-top: 1.6rem;
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
        """,
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
st.session_state.setdefault("adaptive_strength", 50)


def _apply_auto_enhance():
    st.session_state.hist_strength = 55
    st.session_state.freq_strength = 55
    st.session_state.adaptive_strength = 50


def _apply_reset():
    st.session_state.hist_strength = 0
    st.session_state.freq_strength = 0
    st.session_state.adaptive_strength = 0


has_image = st.session_state.source_img is not None

# Both cards live inside a narrower, centered strip rather than stretching
# across the whole wide-mode page — full-width made them read as short,
# flattened bars instead of the more upright cards they're meant to be.
_, center_col, _ = st.columns([1, 7, 1], gap="large")
left_col, right_col = center_col.columns([3, 2], gap="large")

# ------------------------------------------------------------
# LEFT CARD — upload / preview
# ------------------------------------------------------------
with left_col:
    with st.container(border=True, key="upload_card"):
        if not has_image:
            uploaded_file = st.file_uploader(
                "Drop a foggy photo here",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file).convert("RGB")
                st.session_state.source_img = resize_max(np.array(pil_img))
                st.session_state.source_key = f"{uploaded_file.name}-{uploaded_file.size}"
                st.rerun()
        else:
            top_l, top_r = st.columns([3, 1])
            with top_l:
                st.markdown('<div class="card-heading">Preview</div>', unsafe_allow_html=True)
            with top_r:
                if st.button("Change Image", width="stretch"):
                    st.session_state.source_img = None
                    st.session_state.source_key = None
                    st.rerun()

            with st.spinner("Processing…"):
                result = dehaze_pipeline(
                    st.session_state.source_img,
                    hist_strength=st.session_state.hist_strength,
                    freq_strength=st.session_state.freq_strength,
                    adaptive_strength=st.session_state.adaptive_strength,
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
            percent = 0.5 + (hist_strength / 100.0) * 9.5
            if has_image:
                gray = cv2.cvtColor(st.session_state.source_img, cv2.COLOR_RGB2GRAY)
                lo, hi = np.percentile(gray, [percent, 100 - percent])
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
            boost = 0.5 + (freq_strength / 100.0) * 1.5
            cutoff_txt = "30px" if has_image else "n/a"
            st.markdown(
                f'<p class="stage-readout">Boost &times;{boost:.2f} &middot; Cutoff {cutoff_txt}</p>',
                unsafe_allow_html=True,
            )

        # Stage 03 — Adaptive Smoothing + Sharpening
        with st.container(key="stage_adapt"):
            st.markdown(_stage_row("03", "Adaptive Smoothing + Sharpening", st.session_state.adaptive_strength),
                        unsafe_allow_html=True)
            adaptive_strength = st.slider("Adaptive strength", 0, 100, key="adaptive_strength",
                                           label_visibility="collapsed")
            amount = 0.5 + (adaptive_strength / 100.0) * 1.5
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
