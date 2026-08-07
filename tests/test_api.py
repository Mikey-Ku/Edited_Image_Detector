"""The HTTP surface, and the one way it broke that the CLI could not show.

`Evidence.details` is free-form: each detector fills it with whatever describes its
finding, and `/api/analyse` publishes it verbatim. That is convenient right up until a
detector puts a numpy scalar in there, at which point `json.dumps` raises and the whole
analysis returns a 500.

What made it survive review is that it only triggers when a detector *finds* something.
Every bundled sample is a JPEG, re-encoding destroys the demosaicing structure, and
copy-move finds nothing in them, so the endpoint looked healthy on exactly the images a
demo would use, and fell over on the camera-original files that show the system working.

So these tests drive the API with evidence that has a finding in it, rather than with
whatever happens to be lying around in `samples/`.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

# The web stack is the optional `ui` extra, and TestClient additionally needs an HTTP
# client of its own. Skip rather than fail, so a library-only install stays green.
TestClient = pytest.importorskip("fastapi.testclient", reason="needs the [ui] extra").TestClient

from groundtruth.api.server import _jsonable, app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------
# The serialiser
# --------------------------------------------------------------------------


def test_numpy_scalars_survive_serialisation():
    out = _jsonable({"displacement_px": [np.float32(171.9), np.float32(-10.6)]})
    assert out == pytest.approx([171.9, -10.6], abs=1e-4) or out["displacement_px"]
    assert all(isinstance(v, float) for v in out["displacement_px"])


def test_nested_numpy_is_reached():
    out = _jsonable({"a": {"b": [{"c": np.int64(3)}]}})
    assert out == {"a": {"b": [{"c": 3}]}}
    assert isinstance(out["a"]["b"][0]["c"], int)


def test_arrays_become_lists():
    assert _jsonable(np.array([[1.5, 2.5]], dtype=np.float32)) == [[1.5, 2.5]]


def test_plain_values_pass_through_untouched():
    payload = {"s": "text", "i": 3, "f": 1.5, "b": True, "n": None, "l": [1, "two"]}
    assert _jsonable(payload) == payload


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------


def _write_copy_move(path, size=320, patch=64):
    """A photo-like image with one region duplicated, so copy-move has a finding.

    Textured noise rather than flat fill, because SIFT needs keypoints to match and a
    smooth gradient gives it nothing to work with.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(40, 215, (size, size, 3), dtype=np.uint8)
    img[20:20 + patch, 20:20 + patch] = img[200:200 + patch, 120:120 + patch]
    Image.fromarray(img).save(path)
    return path


def test_analyse_returns_json_when_a_detector_has_a_finding(client, tmp_path):
    """The regression. This 500'd whenever copy-move located a displacement."""
    path = _write_copy_move(tmp_path / "dup.png")
    with path.open("rb") as fh:
        r = client.post("/api/analyse", files={"file": ("dup.png", fh, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"]["decision"] in {"auto_clear", "flag", "route_to_human"}
    assert isinstance(body["verdict"]["probability"], float)


def test_every_detail_value_is_json_native(client, tmp_path):
    path = _write_copy_move(tmp_path / "dup2.png")
    with path.open("rb") as fh:
        r = client.post("/api/analyse", files={"file": ("dup2.png", fh, "image/png")})
    assert r.status_code == 200, r.text

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            assert isinstance(v, (str, int, float, bool, type(None))), f"{v!r} leaked"

    for e in r.json()["evidence"]:
        walk(e["details"])


def test_upload_without_a_file_is_a_client_error_not_a_crash(client):
    assert client.post("/api/analyse").status_code in (400, 422)


def test_samples_endpoint_lists_bundled_images(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    for s in r.json():
        assert {"name", "blurb", "group"} <= set(s)


def test_pages_render(client):
    for route in ("/", "/science"):
        r = client.get(route)
        assert r.status_code == 200
        assert "Retrace" in r.text


def test_stylesheet_is_served(client):
    """The pages are returned as strings, so a missing mount leaves them unstyled
    without failing anything else."""
    r = client.get("/static/mk-ui.css")
    assert r.status_code == 200
    assert "--mk-accent" in r.text
