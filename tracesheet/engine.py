from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover - converted into a clear runtime error below
    cv2 = None


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
    flow_window: int = 19
    flow_coherence: float = 0.42
    flow_gap: float = 18.0
    flow_angle: float = 18.0
    colors: int = 8
    region_spatial_radius: int = 12
    region_color_radius: int = 28
    min_region_area: int = 100
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


@dataclass
class SampleSegmentationResult:
    mask: Image.Image
    overlay: Image.Image
    contour: Image.Image
    paths: list[list[tuple[float, float]]]
    accepted_pixels: int


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


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def _merge_small_regions(labels: np.ndarray, minimum_area: int) -> np.ndarray:
    cleaned = labels.astype(np.int32).copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(2):
        changed = False
        for label_id in np.unique(cleaned):
            source = (cleaned == label_id).astype(np.uint8)
            count, components, stats, _centroids = cv2.connectedComponentsWithStats(source, 8)
            for component_id in range(1, count):
                if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum_area:
                    continue
                component = components == component_id
                ring = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool) & ~component
                neighbors = cleaned[ring]
                neighbors = neighbors[neighbors != label_id]
                if neighbors.size:
                    replacement = int(np.bincount(neighbors).argmax())
                    cleaned[component] = replacement
                    changed = True
        if not changed:
            break
    return cleaned


def _segment_regions(image: Image.Image, settings: TraceSettings) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("La modalità a sagome richiede opencv-python-headless.")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.pyrMeanShiftFiltering(
        bgr,
        sp=max(2, int(settings.region_spatial_radius)),
        sr=max(2, int(settings.region_color_radius)),
        maxLevel=1,
    )
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    samples = lab.reshape((-1, 3)).astype(np.float32)
    cluster_count = max(2, min(32, int(settings.colors)))
    cv2.setRNGSeed(0)
    _compactness, labels, centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
        1,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(lab.shape[:2]).astype(np.int32)
    labels = _merge_small_regions(labels, max(1, int(settings.min_region_area)))
    lab_segmented = centers[np.clip(labels, 0, len(centers) - 1)].astype(np.uint8)
    segmented = cv2.cvtColor(lab_segmented, cv2.COLOR_LAB2RGB)
    boundary = _label_boundaries(labels)
    return Image.fromarray(segmented, mode="RGB"), labels, boundary


def _region_paths(labels: np.ndarray, minimum_area: int,
                  epsilon: float) -> list[list[tuple[float, float]]]:
    paths: list[list[tuple[float, float]]] = []
    for label_id in np.unique(labels):
        binary = np.where(labels == label_id, 255, 0).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if abs(float(cv2.contourArea(contour))) < minimum_area:
                continue
            approximated = cv2.approxPolyDP(contour, max(0.1, float(epsilon)), True)
            points = [(float(item[0][0]), float(item[0][1])) for item in approximated]
            if len(points) >= 3:
                points.append(points[0])
                paths.append(points)
    return paths


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


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float64) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92,
                      ((values + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)


def _sample_design(points: np.ndarray, width: int, height: int, linear: bool) -> np.ndarray:
    if not linear:
        return np.ones((len(points), 1), dtype=np.float64)
    return np.column_stack([
        np.ones(len(points)), points[:, 0] / max(1, width - 1),
        points[:, 1] / max(1, height - 1),
    ])


def segment_from_samples(image: Image.Image,
                         positive_points: list[tuple[int, int]],
                         negative_points: list[tuple[int, int]] | None = None,
                         tolerance: float = 0.055,
                         linear_gradient: bool = True,
                         edge_weight: float = 0.35,
                         sample_radius: int = 8,
                         lightness_weight: float = 0.35,
                         chroma_weight: float = 1.0,
                         keep_largest: bool = True,
                         fill_holes: bool = True,
                         close_gaps: int = 2,
                         simplify_pixels: float = 2.0) -> SampleSegmentationResult:
    """Grow a connected OKLab region from user samples using a fitted local gradient."""
    if not positive_points:
        raise ValueError("Aggiungi almeno un campione positivo.")
    if linear_gradient and len(positive_points) < 3:
        raise ValueError("Il gradiente lineare richiede almeno tre campioni positivi.")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    positive = np.asarray(positive_points, dtype=np.int32)
    if np.any(positive[:, 0] < 0) or np.any(positive[:, 0] >= width) or \
            np.any(positive[:, 1] < 0) or np.any(positive[:, 1] >= height):
        raise ValueError("Un campione positivo si trova fuori dall'immagine.")
    oklab = _rgb_to_oklab(rgb)
    # Each click represents a disk, not a fragile single pixel. Trim chromatic
    # outliers inside each disk so dark grain streaks do not drive the model.
    sample_positions: list[tuple[int, int]] = []
    sample_values: list[np.ndarray] = []
    radius = max(0, int(sample_radius))
    yy_disk, xx_disk = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = xx_disk * xx_disk + yy_disk * yy_disk <= radius * radius
    for x, y in positive_points:
        xs = np.clip(x + xx_disk[disk], 0, width - 1)
        ys = np.clip(y + yy_disk[disk], 0, height - 1)
        colors = oklab[ys, xs]
        median = np.median(colors, axis=0)
        deviations = np.linalg.norm(colors - median, axis=1)
        limit = np.quantile(deviations, 0.75) if len(deviations) > 3 else deviations.max(initial=0)
        accepted = deviations <= max(limit, 1e-9)
        sample_positions.extend(zip(xs[accepted].tolist(), ys[accepted].tolist()))
        sample_values.extend(colors[accepted])
    fit_points = np.asarray(sample_positions, dtype=np.float64)
    sample_colors = np.asarray(sample_values, dtype=np.float64)
    design = _sample_design(fit_points, width, height, linear_gradient)
    coefficients, *_ = np.linalg.lstsq(design, sample_colors, rcond=None)
    yy, xx = np.mgrid[0:height, 0:width]
    if linear_gradient:
        expected = coefficients[0] + coefficients[1] * (xx[..., None] / max(1, width - 1)) \
            + coefficients[2] * (yy[..., None] / max(1, height - 1))
    else:
        expected = np.broadcast_to(coefficients[0], oklab.shape)
    delta = oklab - expected
    residual = np.sqrt(
        max(0.0, lightness_weight) * delta[..., 0] ** 2
        + max(0.0, chroma_weight) * (delta[..., 1] ** 2 + delta[..., 2] ** 2)
    )
    luminance = oklab[..., 0]
    gx = np.zeros_like(luminance)
    gy = np.zeros_like(luminance)
    gx[:, 1:] = np.abs(np.diff(luminance, axis=1))
    gy[1:, :] = np.abs(np.diff(luminance, axis=0))
    edge = np.maximum(gx, gy)
    score = residual + max(0.0, edge_weight) * edge
    candidate = score <= max(0.001, tolerance)
    for x, y in positive_points:
        candidate[y, x] = True
    if negative_points:
        for x, y in negative_points:
            if 0 <= x < width and 0 <= y < height:
                candidate[y, x] = False
                negative_radius = max(3, radius)
                candidate[max(0, y-negative_radius):min(height, y+negative_radius+1),
                          max(0, x-negative_radius):min(width, x+negative_radius+1)] = False
    seeds = np.zeros((height + 2, width + 2), dtype=np.uint8)
    flood_source = np.where(candidate, 255, 0).astype(np.uint8)
    connected = np.zeros_like(candidate)
    for x, y in positive_points:
        if connected[y, x]:
            continue
        component = flood_source.copy()
        cv2.floodFill(component, seeds.copy(), (int(x), int(y)), 128)
        connected |= component == 128
    if keep_largest and np.any(connected):
        count, components, stats, _ = cv2.connectedComponentsWithStats(connected.astype(np.uint8), 8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            connected = components == largest
    if close_gaps > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_gaps * 2 + 1,) * 2)
        connected = cv2.morphologyEx(connected.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if fill_holes and np.any(connected):
        inverse = (~connected).astype(np.uint8) * 255
        flood = inverse.copy()
        cv2.floodFill(flood, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 128)
        holes = flood == 255
        connected |= holes
    binary = np.where(connected, 255, 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    paths: list[list[tuple[float, float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        approximated = cv2.approxPolyDP(contour, max(0.1, simplify_pixels), True)
        path = [(float(item[0][0]), float(item[0][1])) for item in approximated]
        if len(path) >= 3:
            path.append(path[0])
            paths.append(path)
    overlay = image.convert("RGB").copy()
    overlay_array = np.asarray(overlay).copy()
    tint = np.array([255, 60, 40], dtype=np.float64)
    overlay_array[connected] = (0.55 * overlay_array[connected] + 0.45 * tint).astype(np.uint8)
    overlay = Image.fromarray(overlay_array)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(overlay)
    for path in paths:
        draw.line(path, fill=(255, 0, 0), width=2, joint="curve")
    contour_image = Image.fromarray(np.where(connected, 0, 255).astype(np.uint8), mode="L")
    return SampleSegmentationResult(
        mask=Image.fromarray(binary, mode="L"), overlay=overlay,
        contour=contour_image, paths=paths, accepted_pixels=int(connected.sum()),
    )


def prepare_raster(image: Image.Image, settings: TraceSettings) -> tuple[Image.Image, np.ndarray, float]:
    work, scale = _resize_for_processing(image.convert("RGB"), settings.max_dimension)
    if settings.mode == "contours":
        raster, _labels, mask = _segment_regions(work, settings)
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
    if settings.mode != "contours":
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


def _path_axis(path: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(path, dtype=float)
    center = values.mean(axis=0)
    if len(values) < 2:
        return center, np.array([1.0, 0.0]), 0.0
    _u, _s, axes = np.linalg.svd(values - center, full_matrices=False)
    direction = axes[0]
    projections = (values - center) @ direction
    return center, direction, float(projections.max(initial=0.0) - projections.min(initial=0.0))


def merge_coherent_paths(paths: list[list[tuple[float, float]]], maximum_gap: float,
                         maximum_angle: float, lateral_tolerance: float = 7.0
                         ) -> list[list[tuple[float, float]]]:
    """Join nearby fragments that belong to one dominant generating direction."""
    groups = [list(path) for path in paths if len(path) >= 2]
    angle_limit = np.deg2rad(max(1.0, maximum_angle))
    changed = True
    while changed:
        changed = False
        best: tuple[float, int, int] | None = None
        for i, first in enumerate(groups):
            center_a, axis_a, _length_a = _path_axis(first)
            for j in range(i + 1, len(groups)):
                second = groups[j]
                center_b, axis_b, _length_b = _path_axis(second)
                angle = np.arccos(np.clip(abs(float(axis_a @ axis_b)), 0.0, 1.0))
                if angle > angle_limit:
                    continue
                delta = center_b - center_a
                lateral = abs(float(delta[0] * axis_a[1] - delta[1] * axis_a[0]))
                if lateral > lateral_tolerance:
                    continue
                endpoints_a = np.asarray([first[0], first[-1]], dtype=float)
                endpoints_b = np.asarray([second[0], second[-1]], dtype=float)
                gap = float(np.linalg.norm(endpoints_a[:, None, :] - endpoints_b[None, :, :], axis=2).min())
                if gap <= maximum_gap and (best is None or gap < best[0]):
                    best = (gap, i, j)
        if best is not None:
            _gap, i, j = best
            combined = groups[i] + groups[j]
            center, direction, _length = _path_axis(combined)
            values = np.asarray(combined, dtype=float)
            projections = (values - center) @ direction
            order = np.argsort(projections)
            groups[i] = [tuple(point) for point in values[order]]
            groups.pop(j)
            changed = True
    result: list[list[tuple[float, float]]] = []
    for group in groups:
        fitted = _fit_line(group, max(lateral_tolerance, 0.5))
        result.append(fitted if fitted is not None else _smooth_curve(_rdp(group, 1.5), 1))
    return result


def _directional_flow_paths(image: Image.Image, settings: TraceSettings
                            ) -> tuple[Image.Image, np.ndarray, list[list[tuple[float, float]]]]:
    """Estimate a structure-tensor field and condense coherent responses into stream paths."""
    if cv2 is None:
        raise RuntimeError("Il motore a flussi richiede opencv-python-headless.")
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), max(0.8, float(settings.blur_radius)))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    window = max(5, int(settings.flow_window) | 1)
    jxx = cv2.boxFilter(gx * gx, -1, (window, window), normalize=True)
    jyy = cv2.boxFilter(gy * gy, -1, (window, window), normalize=True)
    jxy = cv2.boxFilter(gx * gy, -1, (window, window), normalize=True)
    discriminant = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    coherence = discriminant / (jxx + jyy + 1e-6)
    magnitude = cv2.magnitude(gx, gy)
    nonzero = magnitude[magnitude > 0]
    threshold = float(np.percentile(nonzero, 68)) if nonzero.size else 0.0
    candidate = (coherence >= float(settings.flow_coherence)) & (magnitude >= threshold)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8)).astype(bool)
    thin = skeletonize(candidate)
    paths = skeleton_to_paths(thin, settings.min_path_pixels, settings.simplify_pixels)
    paths = merge_coherent_paths(paths, settings.flow_gap, settings.flow_angle)
    raster = Image.fromarray(np.where(candidate, 0, 255).astype(np.uint8), mode="L")
    return raster, thin, paths


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
    if settings.mode == "contours":
        work, scale = _resize_for_processing(original, settings.max_dimension)
        raster, labels, thin = _segment_regions(work, settings)
        paths = _region_paths(labels, settings.min_region_area, settings.simplify_pixels)
        if settings.recognition_mode in {"curves", "hybrid"}:
            paths = [_smooth_curve(path) for path in paths]
    elif settings.recognition_mode == "flows":
        work, scale = _resize_for_processing(original, settings.max_dimension)
        raster, thin, paths = _directional_flow_paths(work, settings)
    else:
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
