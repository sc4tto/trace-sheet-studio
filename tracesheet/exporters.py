from __future__ import annotations

from pathlib import Path

import ezdxf
from PIL import Image

from .engine import TraceResult


def export_raster(image: Image.Image, path: str | Path) -> None:
    image.save(path, format="PNG")


def export_dxf(result: TraceResult, path: str | Path, mm_per_pixel: float = 1.0) -> None:
    """Write a standards-compliant R2010 DXF using closed LWPOLYLINE entities."""
    if not result.paths:
        raise ValueError("Il risultato non contiene contorni vettoriali da esportare.")
    if mm_per_pixel <= 0:
        raise ValueError("La scala in millimetri per pixel deve essere maggiore di zero.")
    source_height = result.source_size[1]
    factor = mm_per_pixel / result.processing_scale
    document = ezdxf.new("R2010", setup=True)
    document.units = ezdxf.units.MM
    if "RICALCO" not in document.layers:
        document.layers.add("RICALCO", color=1)
    modelspace = document.modelspace()
    exported = 0
    for points in result.paths:
        if len(points) < 2:
            continue
        closed = len(points) >= 3 and points[0] == points[-1]
        vertices = points[:-1] if closed else points
        coordinates = [
            (float(x * factor), float((source_height - y / result.processing_scale) * mm_per_pixel))
            for x, y in vertices
        ]
        if len(coordinates) < 2:
            continue
        modelspace.add_lwpolyline(coordinates, close=closed, dxfattribs={"layer": "RICALCO"})
        exported += 1
    if exported == 0:
        raise ValueError("Nessun contorno valido è stato trovato per l'esportazione.")
    document.header["$INSUNITS"] = 4
    document.saveas(Path(path))
