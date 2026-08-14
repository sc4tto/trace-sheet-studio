from pathlib import Path

import numpy as np
import ezdxf
from PIL import Image, ImageDraw

from tracesheet.engine import (TraceResult, TraceSettings, merge_coherent_paths, prepare_raster,
                               segment_from_samples, skeletonize, trace_image)
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
    document = ezdxf.readfile(output)
    assert len(document.modelspace().query("LWPOLYLINE")) >= 1


def test_contour_mode_finds_color_boundaries():
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 80, 60), fill="#7b4422")
    result = trace_image(image, TraceSettings(mode="contours", colors=3, blur_radius=0, min_path_pixels=3))
    assert result.paths
    assert result.raster.mode == "RGB"
    assert result.vector_layers
    assert "02_CURVE_GENERATRICI" in result.vector_layers
    assert all(path[0] == path[-1]
               for path in result.vector_layers["01_TESSERE_CHIUSE"])


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
    document = ezdxf.readfile(output)
    entities = document.modelspace().query("LWPOLYLINE")
    assert len(entities) == 1
    assert entities[0].closed


def test_coherent_fragments_merge_into_one_generator():
    paths = [[(5.0, 20.0), (35.0, 20.5)], [(42.0, 21.0), (80.0, 21.5)]]
    merged = merge_coherent_paths(paths, maximum_gap=12, maximum_angle=8)
    assert len(merged) == 1
    assert len(merged[0]) == 2


def test_directional_flow_mode_returns_vector_paths():
    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 42, 49, 56):
        draw.line((15, y, 165, y + 5), fill="#3b2418", width=2)
    result = trace_image(
        image, TraceSettings(recognition_mode="flows", blur_radius=0.8,
                             min_path_pixels=5, flow_coherence=0.25,
                             flow_gap=20, flow_angle=12),
    )
    assert result.paths
    assert result.overlay.size == image.size


def test_shared_boundaries_export_once_on_separate_layers(tmp_path: Path):
    image = Image.new("RGB", (150, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 49, 89), fill="#e8c999")
    draw.rectangle((50, 0, 99, 89), fill="#86502f")
    draw.rectangle((100, 0, 149, 89), fill="#d89b54")
    result = trace_image(
        image, TraceSettings(mode="contours", colors=3, min_region_area=20,
                             region_merge_delta=2, texture_suppression=3,
                             min_path_pixels=3, simplify_pixels=1),
    )
    assert result.vector_layers
    generators = result.vector_layers["02_CURVE_GENERATRICI"]
    assert 1 <= len(generators) <= 4
    output = tmp_path / "shared.dxf"
    export_dxf(result, output, 1.0)
    document = ezdxf.readfile(output)
    assert "01_TESSERE_CHIUSE" in document.layers
    assert "02_CURVE_GENERATRICI" in document.layers
    assert "03_PERIMETRO" in document.layers
    assert len(document.modelspace().query('*[layer=="02_CURVE_GENERATRICI"]')) == len(generators)


def test_combined_analysis_adds_threshold_lines_to_color_boundaries():
    image = Image.new("RGB", (160, 100), "#c79662")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 0, 159, 99), fill="#714126")
    draw.line((10, 50, 70, 50), fill="black", width=3)
    common = dict(colors=2, min_region_area=20, region_merge_delta=2,
                  texture_suppression=3, min_path_pixels=3,
                  simplify_pixels=1, recognition_mode="hybrid")
    color = trace_image(image, TraceSettings(mode="contours", **common))
    combined = trace_image(
        image, TraceSettings(mode="combined", threshold_mode="manual",
                             automatic_threshold=False, threshold=40,
                             blur_radius=0, close_gaps=0, **common),
    )
    assert len(combined.paths) > len(color.paths)
    assert combined.vector_layers
    assert combined.vector_layers["02_CURVE_GENERATRICI"] == combined.paths
