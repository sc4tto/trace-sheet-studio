from __future__ import annotations

from pathlib import Path

from PIL import Image

from .engine import TraceResult


def export_raster(image: Image.Image, path: str | Path) -> None:
    image.save(path, format="PNG")


def export_dxf(result: TraceResult, path: str | Path, mm_per_pixel: float = 1.0) -> None:
    """Write a small ASCII DXF using standard LWPOLYLINE entities."""
    source_height = result.source_size[1]
    factor = mm_per_pixel / result.processing_scale
    lines = [
        "0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1015",
        "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES", "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
    ]
    for points in result.paths:
        if len(points) < 2:
            continue
        lines.extend(["0", "LWPOLYLINE", "100", "AcDbEntity", "8", "RICALCO",
                      "100", "AcDbPolyline", "90", str(len(points)), "70", "0"])
        for x, y in points:
            lines.extend(["10", f"{x * factor:.6f}", "20", f"{(source_height - y / result.processing_scale) * mm_per_pixel:.6f}"])
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
