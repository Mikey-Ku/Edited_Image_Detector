"""Local web UI for inspecting a single image.

Deliberately a *review* tool, not a verdict tool. The layout puts the per-detector
evidence and its caveats next to the score, because a number without the reasoning
behind it is exactly what an adjuster cannot act on -- and exactly what makes an
automated fraud call dangerous.
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
from PIL import Image

from ..core.types import ClaimContext, ImageCase
from ..fusion.localisation import peak_regions
from ..pipeline.runner import analyse
from ..recovery.reconstruct import reconstruct
from .render import colourise, overlay

app = FastAPI(title="Ground Truth", docs_url="/api/docs")

_STATIC = Path(__file__).parent / "static"

# Bundled synthetic examples so the UI demonstrates itself without the reviewer
# having to go find a manipulated image first.
_SAMPLES = Path(__file__).resolve().parents[3] / "samples"

_SAMPLE_BLURB = {
    "edited_stale_preview.jpg": "Editor rewrote the image but left the EXIF thumbnail — the original is recoverable",
    "splice_noise_mismatch.jpg": "Region spliced in carrying a different sensor-noise level",
    "clean_uniform_noise.jpg": "Unmanipulated control for the noise splice",
    "splice_block_grid.jpg": "Region pasted from another JPEG, carrying a misaligned 8×8 grid",
    "clean_single_grid.jpg": "Unmanipulated control for the block-grid splice",
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "index.html").read_text()


@app.get("/api/samples")
def samples() -> JSONResponse:
    if not _SAMPLES.is_dir():
        return JSONResponse([])
    return JSONResponse(
        [
            {"name": p.name, "blurb": _SAMPLE_BLURB.get(p.name, "")}
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
                        "details": {
                            k: v for k, v in e.details.items() if k != "regions"
                        },
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
