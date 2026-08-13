from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tracesheet.engine import TraceSettings, skeletonize, trace_image
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
