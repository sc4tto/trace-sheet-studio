from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class TraceSettings:
    mode: str = "centerline"
    threshold_mode: str = "otsu"
    threshold: int = 185
    automatic_threshold: bool = True
    sauvola_window: int = 31
    sauvola_k: float = 0.2
    invert: bool = False
    contrast: float = 1.25
    blur_radius: float = 0.6
    close_gaps: int = 1
    min_path_pixels: int = 8
    simplify_pixels: float = 1.5
    recognition_mode: str = "hybrid"
    line_tolerance: float = 1.5
    colors: int = 8
    max_dimension: int = 1400


@dataclass
class TraceResult:
    original: Image.Image
    raster: Image.Image
    skeleton: Image.Image
    overlay: Image.Image
    paths: list[list[tuple[float, float]]]
    processing_scale: float
    source_size: tuple[int, int]


def _otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    weight_bg = 0.0
    sum_bg = 0.0
    best = 127
    best_variance = -1.0
    for value in range(256):
        weight_bg += hist[value]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += value * hist[value]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best = value
    return int(best)


def _resize_for_processing(image: Image.Image, maximum: int) -> tuple[Image.Image, float]:
    largest = max(image.size)
    if largest <= maximum:
        return image.copy(), 1.0
    scale = maximum / largest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def _neighbors(mask: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(mask, 1, mode="constant")
    h, w = mask.shape
    return [
        padded[0:h, 1:w + 1], padded[0:h, 2:w + 2],
        padded[1:h + 1, 2:w + 2], padded[2:h + 2, 2:w + 2],
        padded[2:h + 2, 1:w + 1], padded[2:h + 2, 0:w],
        padded[1:h + 1, 0:w], padded[0:h, 0:w],
    ]


def _dilate(mask: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce([mask, *_neighbors(mask)])


def _erode(mask: np.ndarray) -> np.ndarray:
    return np.logical_and.reduce([mask, *_neighbors(mask)])


def _close(mask: np.ndarray, passes: int) -> np.ndarray:
    result = mask
    for _ in range(max(0, passes)):
        result = _dilate(result)
    for _ in range(max(0, passes)):
        result = _erode(result)
    return result


def _quantized_boundaries(image: Image.Image, colors: int) -> np.ndarray:
    quantized = image.convert("RGB").quantize(colors=max(2, colors), method=Image.Quantize.MEDIANCUT)
    labels = np.asarray(quantized, dtype=np.int16)
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def _box_statistics(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    window = max(3, int(window) | 1)
    radius = window // 2
    values = np.pad(gray.astype(np.float64), radius, mode="reflect")

    def box_sum(array: np.ndarray) -> np.ndarray:
        integral = np.pad(array, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        return integral[window:, window:] - integral[:-window, window:] \
            - integral[window:, :-window] + integral[:-window, :-window]

    area = float(window * window)
    mean = box_sum(values) / area
    variance = np.maximum(0.0, box_sum(values * values) / area - mean * mean)
    return mean, np.sqrt(variance)


def _sauvola(gray: np.ndarray, window: int, k: float) -> np.ndarray:
    mean, deviation = _box_statistics(gray, window)
    local_threshold = mean * (1.0 + float(k) * (deviation / 128.0 - 1.0))
    return gray < local_threshold


def prepare_raster(image: Image.Image, settings: TraceSettings) -> tuple[Image.Image, np.ndarray, float]:
    work, scale = _resize_for_processing(image.convert("RGB"), settings.max_dimension)
    if settings.mode == "contours":
        mask = _quantized_boundaries(work.filter(ImageFilter.GaussianBlur(settings.blur_radius)), settings.colors)
    else:
        gray_image = ImageOps.grayscale(work)
        gray_image = ImageEnhance.Contrast(gray_image).enhance(max(0.05, settings.contrast))
        if settings.blur_radius > 0:
            gray_image = gray_image.filter(ImageFilter.GaussianBlur(settings.blur_radius))
        gray = np.asarray(gray_image, dtype=np.uint8)
        threshold_mode = settings.threshold_mode.lower()
        if threshold_mode == "sauvola":
            mask = _sauvola(gray, settings.sauvola_window, settings.sauvola_k)
        else:
            automatic = settings.automatic_threshold and threshold_mode != "manual"
            threshold = _otsu(gray) if automatic else int(settings.threshold)
            mask = gray < threshold
        if settings.invert:
            mask = ~mask
        mask = _close(mask, settings.close_gaps)
    raster = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L")
    return raster, mask, scale


def skeletonize(mask: np.ndarray, max_iterations: int = 300) -> np.ndarray:
    """Zhang-Suen thinning, implemented in NumPy to keep the desktop build portable."""
    image = mask.astype(bool).copy()
    for _ in range(max_iterations):
        changed = False
        for first in (True, False):
            n = _neighbors(image)
            count = sum(x.astype(np.uint8) for x in n)
            transitions = sum((~n[i] & n[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            common = image & (count >= 2) & (count <= 6) & (transitions == 1)
            if first:
                remove = common & ~(n[0] & n[2] & n[4]) & ~(n[2] & n[4] & n[6])
            else:
                remove = common & ~(n[0] & n[2] & n[6]) & ~(n[0] & n[4] & n[6])
            if np.any(remove):
                image[remove] = False
                changed = True
        if not changed:
            break
    return image


_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _pixel_neighbors(point: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    y, x = point
    return [(y + dy, x + dx) for dy, dx in _OFFSETS if (y + dy, x + dx) in pixels]


def _edge_key(a: tuple[int, int], b: tuple[int, int]):
    return (a, b) if a <= b else (b, a)


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3 or epsilon <= 0:
        return points
    start = np.asarray(points[0], dtype=float)
    end = np.asarray(points[-1], dtype=float)
    values = np.asarray(points, dtype=float)
    line = end - start
    norm = np.linalg.norm(line)
    delta = values - start
    if norm == 0:
        distances = np.linalg.norm(delta, axis=1)
    else:
        # NumPy 2.5 removed the legacy two-dimensional np.cross behavior.
        # The scalar 2D cross product is the signed parallelogram area.
        cross_2d = line[0] * delta[:, 1] - line[1] * delta[:, 0]
        distances = np.abs(cross_2d) / norm
    index = int(np.argmax(distances))
    if distances[index] <= epsilon:
        return [points[0], points[-1]]
    return _rdp(points[:index + 1], epsilon)[:-1] + _rdp(points[index:], epsilon)


def skeleton_to_paths(skeleton: np.ndarray, minimum: int, epsilon: float) -> list[list[tuple[float, float]]]:
    ys, xs = np.nonzero(skeleton)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    if not pixels:
        return []
    degree = {p: len(_pixel_neighbors(p, pixels)) for p in pixels}
    nodes = {p for p, value in degree.items() if value != 2}
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[list[tuple[float, float]]] = []

    def walk(start, nxt):
        path = [start, nxt]
        previous, current = start, nxt
        visited.add(_edge_key(previous, current))
        while current not in nodes:
            candidates = [p for p in _pixel_neighbors(current, pixels)
                          if p != previous and _edge_key(current, p) not in visited]
            if not candidates:
                break
            following = candidates[0]
            path.append(following)
            visited.add(_edge_key(current, following))
            previous, current = current, following
        return path

    for node in sorted(nodes):
        for neighbor in _pixel_neighbors(node, pixels):
            if _edge_key(node, neighbor) in visited:
                continue
            raw = walk(node, neighbor)
            if len(raw) >= minimum:
                points = [(float(x), float(y)) for y, x in raw]
                paths.append(_rdp(points, epsilon))

    # Closed loops have no node; trace any edges that remain.
    for pixel in sorted(pixels):
        for neighbor in _pixel_neighbors(pixel, pixels):
            if _edge_key(pixel, neighbor) in visited:
                continue
            raw = walk(pixel, neighbor)
            if len(raw) >= minimum:
                points = [(float(x), float(y)) for y, x in raw]
                paths.append(_rdp(points, epsilon))
    return paths


def _fit_line(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]] | None:
    if len(points) < 2:
        return None
    values = np.asarray(points, dtype=float)
    center = values.mean(axis=0)
    _, _, axes = np.linalg.svd(values - center, full_matrices=False)
    direction = axes[0]
    projections = (values - center) @ direction
    fitted = center + projections[:, None] * direction
    error = np.linalg.norm(values - fitted, axis=1)
    if float(error.max(initial=0.0)) > max(0.05, tolerance):
        return None
    endpoints = center + np.array([projections.min(), projections.max()])[:, None] * direction
    return [tuple(endpoints[0]), tuple(endpoints[1])]


def _smooth_curve(points: list[tuple[float, float]], iterations: int = 2) -> list[tuple[float, float]]:
    """Chaikin corner cutting: a stable approximation of a quadratic B-spline."""
    if len(points) < 3:
        return points
    current = np.asarray(points, dtype=float)
    for _ in range(max(0, iterations)):
        first = current[:-1]
        second = current[1:]
        q = 0.75 * first + 0.25 * second
        r = 0.25 * first + 0.75 * second
        interior = np.empty((q.shape[0] * 2, 2), dtype=float)
        interior[0::2] = q
        interior[1::2] = r
        current = np.vstack([current[0], interior, current[-1]])
    return [tuple(point) for point in current]


def recognize_paths(paths: list[list[tuple[float, float]]], mode: str,
                    line_tolerance: float) -> list[list[tuple[float, float]]]:
    mode = mode.lower()
    if mode == "centerline":
        return paths
    if mode == "curves":
        return [_smooth_curve(path) for path in paths]
    recognized: list[list[tuple[float, float]]] = []
    for path in paths:
        line = _fit_line(path, line_tolerance)
        if line is not None:
            recognized.append(line)
        elif mode == "hybrid":
            recognized.append(_smooth_curve(path))
    return recognized


def _overlay(original: Image.Image, paths: Iterable[Iterable[tuple[float, float]]], scale: float) -> Image.Image:
    from PIL import ImageDraw

    result = original.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    inverse = 1.0 / scale
    width = max(1, round(2 * inverse))
    for path in paths:
        points = [(x * inverse, y * inverse) for x, y in path]
        if len(points) >= 2:
            draw.line(points, fill=(255, 0, 0), width=width)
    return result


def trace_image(image: Image.Image, settings: TraceSettings) -> TraceResult:
    original = image.convert("RGB")
    raster, mask, scale = prepare_raster(original, settings)
    thin = skeletonize(mask)
    paths = skeleton_to_paths(thin, settings.min_path_pixels, settings.simplify_pixels)
    paths = recognize_paths(paths, settings.recognition_mode, settings.line_tolerance)
    skeleton_image = Image.fromarray(np.where(thin, 0, 255).astype(np.uint8), mode="L")
    return TraceResult(
        original=original,
        raster=raster,
        skeleton=skeleton_image,
        overlay=_overlay(original, paths, scale),
        paths=paths,
        processing_scale=scale,
        source_size=original.size,
    )
