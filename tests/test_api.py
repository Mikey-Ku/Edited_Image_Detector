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


def test_analysing_a_sample_does_not_delete_it(client):
    """Samples are now analysed where they lie, so the residual cache can hit.

    The cleanup in the request handler used to unlink unconditionally, which was
    correct while every request worked on a copy. Against a real path it would
    delete the repository's own sample files, one per page load.
    """
    from groundtruth.api.server import _SAMPLES

    target = _SAMPLES / "real_courtyard_cloned_window.jpg"
    assert target.is_file()
    before = target.stat().st_size
    assert client.post("/api/analyse", data={"sample": target.name}).status_code == 200
    assert target.is_file(), "analysing a sample deleted it"
    assert target.stat().st_size == before


def test_repeat_sample_analysis_is_served_from_cache(client):
    """Second run of the same sample must be much faster than the first.

    Loose bound rather than a tight one, because CI machines vary. The failure this
    guards against is not slowness, it is the caching silently breaking again, which
    costs about eight seconds per example on the landing page.
    """
    import time

    name = "real_courtyard_cloned_window.jpg"
    t0 = time.perf_counter()
    assert client.post("/api/analyse", data={"sample": name}).status_code == 200
    first = time.perf_counter() - t0

    t1 = time.perf_counter()
    assert client.post("/api/analyse", data={"sample": name}).status_code == 200
    second = time.perf_counter() - t1

    assert second < max(first * 0.6, 0.05), (
        f"repeat analysis not cached: first {first:.2f}s, second {second:.2f}s"
    )


def test_sample_row_is_curated_not_a_directory_listing(client):
    """The row is chosen files in a chosen order, not whatever sits in samples/.

    Globbing the directory put the row's contents and its order outside anyone's
    control, and any file dropped in appeared as a button labelled with its own
    filename.
    """
    from groundtruth.api.server import _SAMPLE_ROW, _SAMPLES

    r = client.get("/api/samples")
    assert r.status_code == 200
    row = r.json()

    assert [s["name"] for s in row] == [name for name, _, _ in _SAMPLE_ROW]
    assert len(row) < len(list(_SAMPLES.glob("*.jpg"))), "back to listing the folder"
    for s in row:
        assert {"name", "label", "blurb"} <= set(s)
        assert s["label"] and len(s["label"]) <= 20, f"{s['label']!r} will wrap the row"


def test_the_sample_row_keeps_a_case_retrace_fails(client):
    """A demo row of nothing but successes is a highlight reel.

    Retrace cannot catch a careful hand-composite. real_courtyard_edited.jpg is in
    the row so a visitor can watch it fail on demand, and that button is the
    obvious thing to drop the next time someone tidies the demo up.

    If a detector genuinely improves and this starts flagging, the fix is to
    rewrite the blurb, not to delete the test.
    """
    row = client.get("/api/samples").json()
    miss = next((s for s in row if s["name"] == "real_courtyard_edited.jpg"), None)
    assert miss is not None, "the row no longer offers a case Retrace misses"

    body = client.post("/api/analyse", data={"sample": miss["name"]}).json()
    assert body["verdict"]["decision"] != "flag", (
        "this sample is advertised as one Retrace misses, but it now flags it"
    )


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


# --------------------------------------------------------------------------
# Naming a position in the frame
# --------------------------------------------------------------------------


def test_adjacent_blobs_of_one_edit_get_one_name():
    """The reason this uses fifths rather than thirds.

    On the car sample the damaged strip breaks into two blobs whose centres sit at
    25% and 34% across the frame. Thirds put a boundary at 33%, so they came out
    "left" and "centre": two names for one continuous edit, which reads in the
    table as two separate findings.
    """
    from groundtruth.fusion.localisation import describe_position

    shape = (1200, 1600)
    a = describe_position([328, 702, 474, 778], shape)
    b = describe_position([494, 705, 601, 778], shape)
    assert a == b, f"one edit named two ways: {a!r} and {b!r}"


def test_genuinely_distant_regions_still_separate():
    """Grouping neighbours must not collapse everything into one label."""
    from groundtruth.fusion.localisation import describe_position

    shape = (1200, 1600)
    assert describe_position([328, 702, 474, 778], shape) != describe_position(
        [1078, 722, 1224, 798], shape
    )


def test_the_middle_of_the_frame_is_just_centre():
    """"Middle centre" is not something a person says."""
    from groundtruth.fusion.localisation import describe_position

    assert describe_position([780, 580, 820, 620], (1200, 1600)) == "Centre"


def test_position_names_survive_the_frame_edges():
    """floor(v/n*5) hits 5 exactly at the far edge, which would index off the end."""
    from groundtruth.fusion.localisation import describe_position

    shape = (1200, 1600)
    assert describe_position([0, 0, 0, 0], shape) == "Top far left"
    assert describe_position([1600, 1200, 1600, 1200], shape) == "Bottom far right"


# --------------------------------------------------------------------------
# The proof panel
# --------------------------------------------------------------------------


def test_proof_panel_is_withheld_when_the_evidence_is_weak(client):
    """The panel exists so a person can verify a finding by looking. Two crops only
    marginally more alike than two unrelated patches demonstrate nothing, and a
    number printed beside them reads as proof regardless. Showing nothing is the
    honest output.

    This regressed once already: picking the two hottest heatmap blobs and hoping
    they were the matched pair scored 1.3x on a car panel, which would have shipped.
    """
    from groundtruth.api.server import _SAMPLES

    for name in ("claim_car_original.jpg", "real_courtyard_original.jpg"):
        if not (_SAMPLES / name).is_file():
            continue
        body = client.post("/api/analyse", data={"sample": name}).json()
        assert body.get("proof") is None, f"{name} produced a proof panel with no edit"


def test_proof_panel_reports_a_ratio_worth_believing(client):
    from groundtruth.api.server import _SAMPLES

    name = "claim_car_damage_extended.jpg"
    if not (_SAMPLES / name).is_file():
        pytest.skip("sample not present")
    proof = client.post("/api/analyse", data={"sample": name}).json().get("proof")
    assert proof is not None, "a staged clone should produce a proof panel"
    assert proof["ratio"] >= 3.0
    assert proof["pair_difference"] < proof["control_difference"]


def test_the_proof_panel_says_where_each_crop_came_from(client):
    """Two crops that look the same read as a failed image load, not as evidence.

    The whole argument is that these are *different places* holding *identical
    pixels*, and without the labels only the second half of that is visible. They
    also have to differ: labelling both crops the same place would restate the
    confusion rather than resolve it.
    """
    from groundtruth.api.server import _SAMPLES

    name = "claim_car_damage_extended.jpg"
    if not (_SAMPLES / name).is_file():
        pytest.skip("sample not present")
    proof = client.post("/api/analyse", data={"sample": name}).json()["proof"]
    assert proof["where_a"] and proof["where_b"]
    assert proof["where_a"] != proof["where_b"], "both crops labelled the same place"
