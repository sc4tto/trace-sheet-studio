from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tracesheet.engine import (TraceResult, TraceSettings, prepare_raster, segment_from_samples,
                               skeletonize, trace_image)
from tracesheet.exporters import export_dxf


def test_skeletonize_reduces_thick_line():
    mask = np.zeros((40, 60), dtype=bool)
    mask[16:24, 5:55] = True
    result = skeletonize(mask)
    assert result.sum() < mask.sum()
    assert result.sum() >= 35


def test_centerline_trace_and_dxf(tmp_path: Path):
    image = Image.new("RGB", (160, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((10, 50, 150, 50), fill="black", width=7)
    result = trace_image(image, TraceSettings(automatic_threshold=False, threshold=128, blur_radius=0, close_gaps=0, min_path_pixels=4))
    assert result.paths
    output = tmp_path / "trace.dxf"
    export_dxf(result, output, 0.5)
    text = output.read_text("ascii")
    assert "LWPOLYLINE" in text
    assert "$INSUNITS" in text


def test_contour_mode_finds_color_boundaries():
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 80, 60), fill="#7b4422")
    result = trace_image(image, TraceSettings(mode="contours", colors=3, blur_radius=0, min_path_pixels=3))
    assert result.paths
    assert result.raster.mode == "RGB"
    assert all(path[0] == path[-1] for path in result.paths)


def test_sauvola_handles_uneven_background():
    gradient = np.tile(np.linspace(150, 255, 120, dtype=np.uint8), (80, 1))
    gradient[38:43, 10:110] = 20
    image = Image.fromarray(gradient, mode="L").convert("RGB")
    _raster, mask, _scale = prepare_raster(
        image, TraceSettings(threshold_mode="sauvola", automatic_threshold=False,
                             sauvola_window=21, blur_radius=0, close_gaps=0)
    )
    assert mask[40, 60]
    assert not mask[10, 110]


def test_lines_mode_returns_two_point_entities():
    image = Image.new("RGB", (160, 100), "white")
    ImageDraw.Draw(image).line((10, 50, 150, 50), fill="black", width=7)
    result = trace_image(
        image,
        TraceSettings(threshold_mode="manual", automatic_threshold=False, threshold=128,
                      blur_radius=0, close_gaps=0, min_path_pixels=4,
                      recognition_mode="lines", line_tolerance=2.0),
    )
    assert result.paths
    assert all(len(path) == 2 for path in result.paths)


def test_sample_gradient_grows_connected_region():
    image = np.full((90, 140, 3), 235, dtype=np.uint8)
    for x in range(20, 120):
        shade = 80 + round((x - 20) * 0.7)
        image[20:70, x] = (shade + 30, shade + 10, shade)
    result = segment_from_samples(
        Image.fromarray(image), [(30, 30), (70, 45), (110, 60)],
        [(10, 10)], tolerance=0.08, linear_gradient=True,
    )
    mask = np.asarray(result.mask)
    assert mask[45, 70] == 255
    assert mask[10, 10] == 0
    assert result.paths


def test_sample_segmentation_keeps_largest_and_exports(tmp_path: Path):
    image = np.full((100, 180, 3), 240, dtype=np.uint8)
    image[20:80, 15:75] = (145, 105, 75)
    image[20:80, 110:170] = (145, 105, 75)
    pil = Image.fromarray(image)
    sample = segment_from_samples(
        pil, [(30, 30), (45, 50), (60, 70)], tolerance=0.06,
        linear_gradient=True, sample_radius=4, keep_largest=True, fill_holes=True,
    )
    assert np.asarray(sample.mask)[50, 40] == 255
    assert np.asarray(sample.mask)[50, 140] == 0
    result = TraceResult(pil, sample.mask, sample.contour, sample.overlay,
                         sample.paths, 1.0, pil.size)
    output = tmp_path / "sample.dxf"
    export_dxf(result, output, 1.0)
    assert output.exists()
