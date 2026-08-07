"""Web UI for checking whether a photograph has been edited.

The page explains what it measures before it asks for a file, and every result
shows the reasoning beside the score. A bare probability is what nobody can act
on, and it is what makes an automated call dangerous.
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ..core.types import ClaimContext, ImageCase
from ..fusion.localisation import peak_regions
from ..pipeline.runner import analyse
from ..recovery.reconstruct import reconstruct
from .render import colourise, duplicate_pair, overlay

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warm_examples()
    yield


app = FastAPI(title="Retrace", docs_url="/api/docs", lifespan=_lifespan)

_STATIC = Path(__file__).parent / "static"

# Bundled synthetic examples so the UI demonstrates itself without the reviewer
# having to go find a manipulated image first.
_SAMPLES = Path(__file__).resolve().parents[3] / "samples"

# Served so the worked example on the landing page shows real files rather than
# a mockup of what the tool would produce.
# (blurb, group). Real photographs are grouped separately and labelled honestly:
# Retrace currently cannot separate them from their own unedited originals, and
# hiding that behind a flattering demo would misrepresent the tool.
_SAMPLE_INFO = {
    "claim_car_damage_extended.jpg": (
        ("A real damaged car, with a crumpled section of the front door cloned onto "
         "a rear panel that was never hit. No original attached: the image is caught "
         "disagreeing with itself."), "single"),
    "claim_wall_crack_duplicated.jpg": (
        ("A real subsidence crack, partly duplicated so the damage looks worse. The "
         "file also still carries a thumbnail of the pre-edit original."), "single"),
    "claim_car_original.jpg": (
        "The untouched car photograph, for checking the system's work.", "real"),
    "claim_wall_original.jpg": (
        "The untouched wall photograph, for checking the system's work.", "real"),
    "edited_stale_preview.jpg": (
        ("Synthetic. The editor rewrote the image but left the embedded preview, "
         "so the original is recoverable."), "synthetic"),
    "splice_noise_mismatch.jpg": (
        "Synthetic. A region spliced in carrying a different sensor-noise level.", "synthetic"),
    "clean_uniform_noise.jpg": (
        "Synthetic. Unedited control for the noise splice.", "synthetic"),
    "real_courtyard_cloned_window.jpg": (
        "A decorative window clone-stamped along a wall. Caught from the image alone.",
        "single"),
    "real_courtyard_edited.jpg": (
        ("Real photograph, two figures composited in by hand in GIMP. "
         "Retrace does not detect this one."), "real"),
    "real_courtyard_original.jpg": (
        "The unedited original of the courtyard photograph.", "real"),
    "real_courtyard_stale_preview.jpg": (
        ("Real photograph, hand-edited, and the file still holds a preview of the "
         "original. Retrace recovers it."), "real"),
    "real_rooftops_stale_preview.jpg": (
        ("Real photograph, hand-edited, preview of the original intact. "
         "Retrace recovers it."), "real"),
}

# Panels are for looking at, not for measuring. Downscaling keeps the payload
# reasonable on a 1920x1080 source; all analysis runs at native resolution.
_MAX_PANEL_PX = 900


def _fit(arr: np.ndarray) -> Image.Image:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if max(img.size) > _MAX_PANEL_PX:
        scale = _MAX_PANEL_PX / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return img


def _data_uri(arr: np.ndarray | None) -> str | None:
    if arr is None:
        return None
    buf = io.BytesIO()
    _fit(arr).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _jsonable(value):
    """Coerce numpy scalars and arrays into plain Python for the JSON response.

    `Evidence.details` is a free-form dict every detector fills in however suits it,
    and it is published verbatim. That is a good arrangement for detector authors and
    a sharp edge for this endpoint: a stray `np.float32` raises inside `json.dumps`
    and turns the whole analysis into a 500, with the traceback pointing at the
    serialiser rather than at the detector that produced it.

    Detectors should still emit plain types, and `geometric.copy_move` now does. This
    is the belt to that pair of braces, because the failure only appears when a
    detector *finds* something, which is the one case nobody demos before shipping.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


# Rendered results for the bundled samples, keyed by (name, mtime, claim context).
# The sample files never change between requests, so recomputing their analysis on
# every page load buys nothing: the landing page runs two of them before it can show
# anything, and that was the whole of its load time. Uploads are never cached, since
# each one is genuinely new.
_SAMPLE_CACHE: dict[tuple, dict] = {}
_SAMPLE_CACHE_MAX = 16


def _cached_sample(key: tuple) -> dict | None:
    return _SAMPLE_CACHE.get(key)


def _store_sample(key: tuple, payload: dict) -> None:
    if len(_SAMPLE_CACHE) >= _SAMPLE_CACHE_MAX:
        _SAMPLE_CACHE.pop(next(iter(_SAMPLE_CACHE)))
    _SAMPLE_CACHE[key] = payload


_CACHE_LOCK = threading.Lock()
_INFLIGHT: dict[tuple, threading.Lock] = {}


def _analyse_sample(key: tuple, path: Path, display_name: str, context) -> dict:
    """Cached sample analysis, computed at most once per key even under concurrency.

    Without the lock this is a thundering herd, and measurably so. The startup
    warm-up and the page's own two requests all arrive within a second of each
    other, all miss the empty cache, and all start the same torch inference: three
    copies of the same eight-second job competing for the same cores. Measured that
    way, the second example took **thirty seconds** to return, far worse than having
    no warm-up at all.

    Double-checked around the lock, so the winner computes and everyone else picks
    up the finished result instead of repeating it.
    """
    hit = _cached_sample(key)
    if hit is not None:
        return hit

    with _CACHE_LOCK:
        lock = _INFLIGHT.setdefault(key, threading.Lock())

    with lock:
        hit = _cached_sample(key)
        if hit is not None:
            return hit
        payload = _build_payload(path, display_name, context)
        _store_sample(key, payload)

    with _CACHE_LOCK:
        _INFLIGHT.pop(key, None)
    return payload


# The two samples the landing page analyses before it can render anything.
_WARM_ON_START = ("claim_car_damage_extended.jpg", "claim_wall_crack_duplicated.jpg")


def _warm_examples() -> None:
    """Analyse the landing-page examples in the background as the server comes up.

    Caching alone only helps the second visitor. The first one still waits through
    two full analyses, roughly eight seconds each on a 1920x1080 frame, staring at
    empty panels. Doing that work at startup moves the wait to a moment when nobody
    is looking.

    On a thread, and deliberately silent on failure: this is an optimisation, and a
    server that refuses to start because a sample is missing would be a worse
    outcome than a slow first page.
    """
    def run() -> None:
        for name in _WARM_ON_START:
            try:
                path = _SAMPLES / name
                if not path.is_file():
                    continue
                key = (name, path.stat().st_mtime, "", "", "unknown", "unknown")
                _analyse_sample(key, path, name, None)
            except Exception:
                log.debug("could not pre-warm %s", name, exc_info=True)

    if _SAMPLES.is_dir():
        threading.Thread(target=run, name="warm-examples", daemon=True).start()


if _SAMPLES.is_dir():
    app.mount("/samples", StaticFiles(directory=str(_SAMPLES)), name="samples")

# The pages are read and returned as strings rather than served from here, but the
# stylesheet they pull in has to be fetchable. Mounted rather than inlined so the
# vendored mk-ui.css stays a recognisable copy of its upstream file.
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "index.html").read_text()


@app.get("/science", response_class=HTMLResponse)
def science() -> str:
    return (_STATIC / "science.html").read_text()


@app.get("/api/samples")
def samples() -> JSONResponse:
    if not _SAMPLES.is_dir():
        return JSONResponse([])
    return JSONResponse(
        [
            {
                "name": p.name,
                "blurb": _SAMPLE_INFO.get(p.name, ("", "synthetic"))[0],
                "group": _SAMPLE_INFO.get(p.name, ("", "synthetic"))[1],
            }
            for p in sorted(_SAMPLES.glob("*.jpg"))
        ]
    )


@app.post("/api/analyse")
async def analyse_upload(
    file: UploadFile | None = File(None),
    sample: str = Form(""),
    claim_id: str = Form("unknown"),
    claimant_id: str = Form("unknown"),
    policy_inception: str = Form(""),
    loss_date: str = Form(""),
) -> JSONResponse:
    if sample:
        # Resolve against the samples directory and verify containment, so a
        # crafted name cannot walk out of it and read arbitrary files.
        candidate = (_SAMPLES / sample).resolve()
        if not candidate.is_file() or _SAMPLES.resolve() not in candidate.parents:
            raise HTTPException(404, f"no such sample: {sample}")
        # Analysed where it lies, rather than copied to a temp file first. The
        # noiseprint residual is memoised on (path, quality, mtime), and a fresh
        # random temp name every request meant that cache could never hit: the two
        # landing-page examples paid full torch inference on every page load, about
        # eight seconds each. A stable path makes repeat loads effectively free.
        path, display_name, temporary = candidate, candidate.name, False
        cache_key = (
            candidate.name, candidate.stat().st_mtime,
            policy_inception, loss_date, claim_id, claimant_id,
        )

    elif file is not None:
        raw = await file.read()
        display_name = file.filename or "upload"
        if not raw:
            raise HTTPException(400, "empty upload")
        with tempfile.NamedTemporaryFile(
            suffix=Path(display_name).suffix or ".jpg", delete=False
        ) as tmp:
            tmp.write(raw)
            path = Path(tmp.name)
        temporary = True
    else:
        raise HTTPException(400, "provide either a file upload or a sample name")

    inception = _parse_date(policy_inception)
    loss = _parse_date(loss_date)
    context = (
        ClaimContext(
            claim_id=claim_id,
            claimant_id=claimant_id,
            policy_inception=inception,
            loss_date=loss,
        )
        if (inception or loss)
        else None
    )

    try:
        if temporary:
            payload = _build_payload(path, display_name, context)
        else:
            payload = _analyse_sample(cache_key, path, display_name, context)
        return JSONResponse(payload)
    finally:
        # Only ever delete what this request created. Deleting unconditionally would
        # now remove a bundled sample from the repository.
        if temporary:
            path.unlink(missing_ok=True)


def _build_payload(
    path: Path, display_name: str, context: ClaimContext | None = None
) -> dict:
    """Run the pipeline over one file and render everything the UI shows.

    Split out of the endpoint so the startup warm-up can call it directly instead of
    making an HTTP request to its own process.
    """
    case = ImageCase(image_path=path, context=context)
    verdict = analyse(case)
    base = case.pixels()

    images: dict[str, str | None] = {"original": _data_uri(base)}
    regions: list[dict] = []
    proof: dict | None = None
    if verdict.heatmap is not None:
        images["overlay"] = _data_uri(overlay(base, verdict.heatmap))
        images["heatmap"] = _data_uri(colourise(verdict.heatmap))
        regions = peak_regions(verdict.heatmap, threshold=0.5)

        # When a region was flagged, the finding can be shown rather than asserted:
        # crop it and its partner and put them side by side. Absent for most images,
        # which is why the UI treats it as an extra panel and not a fixture.
        #
        # The partner is located by copy-move's own measured displacement rather than
        # by guessing from the heatmap, so the pair shown is the pair the detector
        # actually matched.
        cm = next((e for e in verdict.evidence
                   if e.detector_id == "geometric.copy_move" and e.applicable), None)
        disp = (cm.details or {}).get("displacement_px") if cm else None
        made = duplicate_pair(
            base, regions,
            displacement=tuple(disp) if disp and len(disp) == 2 else None,
        )
        if made is not None:
            pair_img, proof = made
            images["proof"] = _data_uri(pair_img)

    recovery = None
    recon = reconstruct(path)
    if recon is not None:
        recovery = {
            "fidelity": recon.fidelity.value,
            "is_evidence": recon.is_evidence,
            "source": recon.source,
            "preview_size": list(recon.preview_size),
            "changed_fraction": round(recon.changed_fraction, 4),
            "cropped": recon.cropped,
            "caveat": recon.caveat,
            "regions": recon.regions[:5],
        }
        images["before"] = _data_uri(recon.before)
        images["changed"] = _data_uri(colourise(recon.difference))

    return {
        "filename": display_name,
        "container": {
            "actual": case.container.actual.value,
            "claimed": case.container.claimed.value,
            "mismatch": case.container.extension_mismatch,
            "lossy": case.container.actual.lossy,
        },
        "size": [int(base.shape[1]), int(base.shape[0])],
        "verdict": {
            "probability": round(verdict.manipulated_probability, 4),
            "decision": verdict.decision.value,
            "explanation": verdict.explanation,
            "localised_by": verdict.localised_by,
        },
        "evidence": [
            {
                "detector_id": e.detector_id,
                "tier": e.tier.value,
                "applicable": e.applicable,
                "score": round(e.score, 4),
                "confidence": round(e.confidence, 4),
                "explanation": e.explanation,
                "localises": e.heatmap is not None,
                "details": _jsonable(
                    {k: v for k, v in e.details.items() if k != "regions"}
                ),
            }
            for e in verdict.evidence
        ],
        "regions": regions[:8],
        "proof": proof,
        "recovery": recovery,
        "images": images,
    }
