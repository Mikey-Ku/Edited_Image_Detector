"""Web UI for checking whether a photograph has been edited.

The page explains what it measures before it asks for a file, and every result
shows the reasoning beside the score. A bare probability is what nobody can act
on, and it is what makes an automated call dangerous.
"""

from __future__ import annotations

import base64
import io
import tempfile
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
from .render import colourise, overlay

app = FastAPI(title="Retrace", docs_url="/api/docs")

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
    "real_courtyard_cloned_window.jpg": (
        ("Real photograph with one decorative window clone-stamped along the wall. "
         "No embedded preview, so nothing is compared against an original: the image "
         "is caught disagreeing with itself."), "single"),
    "edited_stale_preview.jpg": (
        ("Synthetic. The editor rewrote the image but left the embedded preview, "
         "so the original is recoverable."), "synthetic"),
    "splice_noise_mismatch.jpg": (
        "Synthetic. A region spliced in carrying a different sensor-noise level.", "synthetic"),
    "clean_uniform_noise.jpg": (
        "Synthetic. Unedited control for the noise splice.", "synthetic"),
    "real_courtyard_edited.jpg": (
        ("Real photograph (Nikon D7000), two figures composited in by hand in GIMP. "
         "Retrace does not currently detect this."), "real"),
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
        raw = candidate.read_bytes()
        display_name = candidate.name
    elif file is not None:
        raw = await file.read()
        display_name = file.filename or "upload"
        if not raw:
            raise HTTPException(400, "empty upload")
    else:
        raise HTTPException(400, "provide either a file upload or a sample name")

    with tempfile.NamedTemporaryFile(
        suffix=Path(display_name).suffix or ".jpg", delete=False
    ) as tmp:
        tmp.write(raw)
        path = Path(tmp.name)

    try:
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

        case = ImageCase(image_path=path, context=context)
        verdict = analyse(case)
        base = case.pixels()

        images: dict[str, str | None] = {"original": _data_uri(base)}
        regions: list[dict] = []
        if verdict.heatmap is not None:
            images["overlay"] = _data_uri(overlay(base, verdict.heatmap))
            images["heatmap"] = _data_uri(colourise(verdict.heatmap))
            regions = peak_regions(verdict.heatmap, threshold=0.5)

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

        return JSONResponse(
            {
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
                "recovery": recovery,
                "images": images,
            }
        )
    finally:
        path.unlink(missing_ok=True)
