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

    Returns (html, row_width_px) — the caller needs the exact width to know
    how many times this row must repeat to safely outrun any viewport
    (see `_lane_html`).
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
        top = f"margin-top:{rnd.randint(*top_range)}px;" if top_range else ""
        parts.append(
            f'<img class="sprite" style="height:{h}px;{top}margin-right:{gap}px;" '
            f'src="data:image/png;base64,{b64}">'
        )
        total_w += h * aspect + gap
    return "".join(parts), total_w


MIN_TRACK_WIDTH = 8000  # px a lane must cover before it's safe on any realistic viewport
# Streamlit's own page padding around the main content block is a fixed
# ~80px per side regardless of window size (verified empirically), and the
# header strip itself now runs full-bleed to the true browser edges (see
# inject_css), so this has to cover the *entire* window width on a large
# display — comfortably over 5000px on a 5K screen — with margin to spare.


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


def render_header():
    cars = load_sprite_meta("cars")
    clouds = load_sprite_meta("clouds")

    if not cars or not clouds:
        # Graceful fallback if the assets folder isn't next to the script.
        st.title("🌫️ Urban Fog Cleaner")
        st.caption(
            "Drop in a hazy photo or pull a frame from a camera export. "
            "Everything runs on this machine through the real dark-channel-prior "
            "dehazing pipeline from urban.py."
        )
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
        _lane_html(clouds, count=9, height_range=(75, 110), gap_range=(15, 35),
                   seed=7, duration=44, top_range=(15, 45), opacity=0.95)
        + _lane_html(clouds, count=9, height_range=(55, 85), gap_range=(13, 28),
                     seed=51, duration=61, top_range=(35, 65), opacity=0.80)
        + _lane_html(clouds, count=9, height_range=(40, 60), gap_range=(11, 24),
                     seed=83, duration=79, top_range=(55, 85), opacity=0.62)
    )

    html = f"""
    <div class="ufc-header">
      <div class="ufc-topbar">Urban Fog Cleaner</div>
      <div class="ufc-hero">
        <h1>Load a Frame</h1>
        <p>Drop in a hazy photo or pull a frame from a camera export. Everything runs
        on this machine through the real dark-channel-prior dehazing pipeline &mdash;
        histogram stretching, frequency-domain detail boost, and adaptive
        smoothing/sharpening, all tunable live.</p>
      </div>
      <div class="ufc-strip">
        {cloud_layers_html}
        {car_layers_html}
      </div>
    </div>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
      * {{ box-sizing:border-box; }}
      html, body {{ margin:0; background:#f5f5f5; font-family:'IBM Plex Sans',sans-serif; }}
      .ufc-header {{ position:relative; overflow:hidden; background:#f5f5f5; }}
      /* The iframe itself is now full-bleed (see inject_css), so the
         topbar/title get their own left/right padding here to line back
         up with the cards below — which now sit in a narrower, centered
         strip (see the st.columns([1,7,1]) wrapper below the header),
         not flush against the page edge. That wrapper's left edge works
         out to (100vw/9 + 84px) — a spacer column that's 1/9 of the
         content width plus Streamlit's own fixed page margin — so this
         mirrors that formula instead of a fixed pixel value, to keep
         tracking it as the window is resized. */
      .ufc-topbar {{
        font-family:'Big Shoulders Display',sans-serif; font-weight:800; font-size:19px;
        text-transform:uppercase; letter-spacing:.03em; color:#111;
        padding:20px calc(100vw/9 + 84px) 0 calc(100vw/9 + 84px);
      }}
      .ufc-hero {{
        padding:4px calc(100vw/9 + 84px) 18px calc(100vw/9 + 84px);
        max-width:calc(856px + 100vw/9);
      }}
      .ufc-hero h1 {{
        font-family:'Big Shoulders Display',sans-serif; font-weight:800; font-size:56px;
        text-transform:uppercase; margin:2px 0 8px 0; line-height:0.95; color:#111;
      }}
      .ufc-hero p {{ font-size:14.5px; color:#4a4a4a; line-height:1.55; margin:0; }}
      .ufc-strip {{
        position:relative; height:170px; overflow:hidden;
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
      .cloud-layer {{ align-items:flex-start; padding-top:6px; }}
      .car-layer {{ align-items:flex-end; padding-bottom:12px; }}
      img.sprite {{ image-rendering:pixelated; display:block; flex-shrink:0; }}
      @keyframes scrollLTR {{
        from {{ transform:translateX(var(--shift)); }}
        to   {{ transform:translateX(0%); }}
      }}
    </style>
    """
    _embed_html(html, height=440)


# ============================================================
# PAGE-WIDE CSS (fonts + card typography, no widget-overlay hacks)
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        [data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]) {
            font-family:'IBM Plex Sans', sans-serif;
        }

        /* Break the header iframe out of Streamlit's centered content
           column so the cars/clouds strip can run edge-to-edge of the
           actual browser window. The rest of the page (upload card,
           pipeline panel) is untouched and keeps its normal margins —
           only the element wrapping this one iframe is affected.
           !important is needed because Streamlit sets an explicit width
           on this element that would otherwise win. */
        div[data-testid="stElementContainer"]:has(iframe[data-testid="stIFrame"]) {
            width: 100vw !important;
            max-width: 100vw !important;
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw !important;
            margin-right: -50vw !important;
        }

        /* Make the upload dropzone bigger and center its contents */
        [data-testid="stFileUploaderDropzone"] {
            min-height: 400px;
            padding: 36px 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 14px;
            border: 2px dashed rgba(0,0,0,.18);
            border-radius: 12px;
            background: #fafafa;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        /* Red, chunkier "Upload" button inside the dropzone */
        [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
            background-color: #d32f2f;
            border-color: #d32f2f;
            color: #ffffff;
            padding: 0.6rem 1.6rem;
            font-size: 15px;
            border-radius: 6px;
        }
        [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #b71c1c;
            border-color: #b71c1c;
            color: #ffffff;
        }

        /* Give both cards real vertical padding so they read as upright
           panels instead of thin horizontal strips (they're targeted by
           the container `key=`, so this can't leak onto other borders). */
        .st-key-upload_card, .st-key-pipeline_card {
            padding: 28px 26px 32px 26px;
        }
        .st-key-pipeline_card {
            display: flex;
            flex-direction: column;
        }

        .pipeline-heading {
            font-family:'Big Shoulders Display', sans-serif; text-transform:uppercase;
            font-weight:800; font-size:19px; letter-spacing:.02em; color:#111;
            border-bottom:1px solid rgba(0,0,0,.08); padding-bottom:10px; margin-bottom:6px;
        }
        .card-heading {
            font-family:'Big Shoulders Display', sans-serif; text-transform:uppercase;
            font-weight:800; font-size:19px; letter-spacing:.02em; color:#111; margin-bottom:2px;
        }
        .stage-row {
            display:flex; justify-content:space-between; align-items:baseline;
            margin-top:16px;
        }
        .stage-num {
            font-family:'IBM Plex Mono', monospace; color:#d32f2f; font-size:12px;
            margin-right:8px;
        }
        .stage-label { font-weight:600; font-size:14px; color:#111; }
        .stage-value {
            font-family:'IBM Plex Mono', monospace; color:#d32f2f; font-weight:600; font-size:14px;
        }
        .stage-readout {
            font-family:'IBM Plex Mono', monospace; font-size:11px; color:#888888; margin:2px 0 0 0;
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

            col_before, col_after = st.columns(2)
            with col_before:
                st.image(st.session_state.source_img, caption="Original", width="stretch")
            with col_after:
                st.image(result, caption="Dehazed", width="stretch")

# ------------------------------------------------------------
# RIGHT CARD — enhancement pipeline
# ------------------------------------------------------------
with right_col:
    with st.container(border=True, key="pipeline_card"):
        st.markdown('<div class="pipeline-heading">Enhancement Pipeline</div>', unsafe_allow_html=True)

        # Stage 01 — Histogram Stretch
        hv = st.session_state.hist_strength
        st.markdown(
            f'<div class="stage-row"><span><span class="stage-num">01</span>'
            f'<span class="stage-label">Histogram Stretch</span></span>'
            f'<span class="stage-value">{hv}</span></div>',
            unsafe_allow_html=True,
        )
        hist_strength = st.slider("Histogram strength", 0, 100, key="hist_strength",
                                   label_visibility="collapsed")
        if has_image:
            percent = 0.5 + (hist_strength / 100.0) * 9.5
            gray = cv2.cvtColor(st.session_state.source_img, cv2.COLOR_RGB2GRAY)
            lo, hi = np.percentile(gray, [percent, 100 - percent])
            st.markdown(
                f'<p class="stage-readout">Clip {percent:.1f}% &middot; Range {int(lo)}&ndash;{int(hi)}</p>',
                unsafe_allow_html=True,
            )
        else:
            percent = 0.5 + (hist_strength / 100.0) * 9.5
            st.markdown(
                f'<p class="stage-readout">Clip {percent:.1f}% &middot; Range n/a</p>',
                unsafe_allow_html=True,
            )

        # Stage 02 — Frequency-Domain Filter
        fv = st.session_state.freq_strength
        st.markdown(
            f'<div class="stage-row"><span><span class="stage-num">02</span>'
            f'<span class="stage-label">Frequency-Domain Filter</span></span>'
            f'<span class="stage-value">{fv}</span></div>',
            unsafe_allow_html=True,
        )
        freq_strength = st.slider("Frequency strength", 0, 100, key="freq_strength",
                                   label_visibility="collapsed")
        boost = 0.5 + (freq_strength / 100.0) * 1.5
        cutoff_txt = "30px" if has_image else "n/a"
        st.markdown(
            f'<p class="stage-readout">Boost &times;{boost:.2f} &middot; Cutoff {cutoff_txt}</p>',
            unsafe_allow_html=True,
        )

        # Stage 03 — Adaptive Smoothing + Sharpening
        av = st.session_state.adaptive_strength
        st.markdown(
            f'<div class="stage-row"><span><span class="stage-num">03</span>'
            f'<span class="stage-label">Adaptive Smoothing + Sharpening</span></span>'
            f'<span class="stage-value">{av}</span></div>',
            unsafe_allow_html=True,
        )
        adaptive_strength = st.slider("Adaptive strength", 0, 100, key="adaptive_strength",
                                       label_visibility="collapsed")
        amount = 0.5 + (adaptive_strength / 100.0) * 1.5
        st.markdown(
            f'<p class="stage-readout">Edge gain &times;{amount:.2f}</p>',
            unsafe_allow_html=True,
        )

        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            st.button("Auto-Enhance", width="stretch", on_click=_apply_auto_enhance)
        with b2:
            st.button("Reset", width="stretch", on_click=_apply_reset)

        st.write("")
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
