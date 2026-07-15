"""
src/preprocessing/segmentation.py

Iris segmentation and normalization pipeline for IrisNet.

Pipeline (in order):
  1. denoise_image    — load + median blur + gaussian blur
  2. segment_iris     — two-phase HoughCircles (pupil then iris) with fallback
  3. normalize_iris   — Daugman Rubber-Sheet Model via cv2.remap
  4. scale_pixels     — resize to (128,128), float32 [0,1], expand to (128,128,1)
"""

import cv2
import numpy as np
from typing import Optional


DEFAULT_SEGMENTATION_CONFIG = {
    "pupil": {
        "dp": 1.0,
        "min_dist": 50,
        "param1": 100,
        "param2_start": 50,
        "param2_min": 5,
        "param2_step": -5,
        "min_radius": 10,
        "max_radius": 80,
    },
    "iris": {
        "dp": 1.0,
        "min_dist": 50,
        "param1": 100,
        "param2_start": 30,
        "param2_min": 5,
        "param2_step": -5,
        "min_radius": 80,
        "max_radius": 200,
    },
    "center_offset_frac": 0.60,
    "center_offset_floor": 60.0,
}


def denoise_image(image_path: str) -> np.ndarray:
    """Load a grayscale iris image and apply noise reduction.

    Applies median blur first to suppress salt-and-pepper noise and specular
    reflections (common in CASIA-Iris-Lamp), then Gaussian blur to smooth
    remaining high-frequency noise.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the .jpg iris image.

    Returns
    -------
    np.ndarray
        Denoised grayscale image as a uint8 numpy array.

    Raises
    ------
    FileNotFoundError
        If the image cannot be loaded from image_path.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.medianBlur(image, ksize=5)
    image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1.5)
    return image


def _detect_all(image, cfg):
    """Run HoughCircles with a fallback loop on param2; return circles or None."""
    for p2 in range(cfg["param2_start"], cfg["param2_min"] - 1, cfg["param2_step"]):
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=cfg["dp"],
            minDist=cfg["min_dist"],
            param1=cfg["param1"],
            param2=p2,
            minRadius=cfg["min_radius"],
            maxRadius=cfg["max_radius"],
        )
        if circles is not None:
            return np.round(circles[0, :]).astype(int)
    return None


def _merge_segmentation_config(config: Optional[dict]) -> dict:
    """Merge a partial segmentation config with the production defaults."""
    merged = {
        "pupil": DEFAULT_SEGMENTATION_CONFIG["pupil"].copy(),
        "iris": DEFAULT_SEGMENTATION_CONFIG["iris"].copy(),
        "center_offset_frac": DEFAULT_SEGMENTATION_CONFIG["center_offset_frac"],
        "center_offset_floor": DEFAULT_SEGMENTATION_CONFIG["center_offset_floor"],
    }
    if not config:
        return merged
    for key in ("pupil", "iris"):
        if key in config:
            merged[key].update(config[key])
    for key in ("center_offset_frac", "center_offset_floor"):
        if key in config:
            merged[key] = config[key]
    return merged


def validate_iris_circles(circles: dict, image_shape: tuple) -> tuple:
    """Validate circle geometry and return (is_valid, reason)."""
    h, w = image_shape[:2]
    cx, cy = circles["center"]
    r_pupil = circles["r_pupil"]
    r_iris = circles["r_iris"]

    if r_pupil <= 0 or r_iris <= 0:
        return False, "non_positive_radius"
    if r_iris <= r_pupil:
        return False, "iris_not_larger_than_pupil"
    if not (0 <= cx < w and 0 <= cy < h):
        return False, "center_outside_image"
    if r_iris > max(h, w):
        return False, "iris_radius_too_large"
    return True, "ok"


def segment_iris_configurable(blurred_image: np.ndarray,
                              config: Optional[dict] = None) -> Optional[dict]:
    """Detect pupil and iris boundaries using two-phase HoughCircles.

    Performs two independent Hough circle detections:
      - Phase 1: pupil (inner boundary) — small, dark circle
      - Phase 2: iris  (outer boundary) — larger circle surrounding the pupil

    Each phase uses a fallback loop that progressively relaxes param2
    (circle accumulator threshold) until circles are found.

    A sanity check validates that r_iris > r_pupil and that the two centres
    are within 25 pixels of each other.

    Parameters
    ----------
    blurred_image : np.ndarray
        Denoised grayscale image (output of denoise_image).

    Returns
    -------
    dict or None
        On success: {"center": (cx, cy), "r_pupil": float, "r_iris": float}
        On failure (no valid circles found): None
    """
    cfg = _merge_segmentation_config(config)
    h, w = blurred_image.shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0   # image centre — used as prior

    # --- Phase 1: Pupil (select circle closest to image centre) ---
    pupil_candidates = _detect_all(blurred_image, cfg["pupil"])
    if pupil_candidates is None:
        return None

    dists_to_centre = np.sqrt(
        (pupil_candidates[:, 0] - img_cx) ** 2 +
        (pupil_candidates[:, 1] - img_cy) ** 2
    )
    px, py, r_pupil = pupil_candidates[np.argmin(dists_to_centre)]

    # --- Phase 2: Iris (select circle closest to pupil centre) ---
    # maxRadius=200 covers both 280x320 (r~90-130) and 480x640 (r~100-180) images
    iris_candidates = _detect_all(blurred_image, cfg["iris"])
    if iris_candidates is None:
        return None

    dists_to_pupil = np.sqrt(
        (iris_candidates[:, 0] - px) ** 2 +
        (iris_candidates[:, 1] - py) ** 2
    )
    ix, iy, r_iris = iris_candidates[np.argmin(dists_to_pupil)]

    # --- Sanity check ---
    # Allow centre offset up to 60% of iris radius (handles Lamp illumination variance
    # where the ring illuminator shifts apparent iris centre), floor of 60 px.
    center_dist = float(np.sqrt((px - ix) ** 2 + (py - iy) ** 2))
    max_offset = max(cfg["center_offset_frac"] * r_iris, cfg["center_offset_floor"])
    if r_iris <= r_pupil or center_dist > max_offset:
        return None

    # Use pupil centre as the canonical centre (more stable)
    circles = {
        "center": (int(px), int(py)),
        "r_pupil": float(r_pupil),
        "r_iris": float(r_iris),
        "pupil_center": (int(px), int(py)),
        "iris_center": (int(ix), int(iy)),
        "center_distance": center_dist,
    }
    ok, _ = validate_iris_circles(circles, blurred_image.shape)
    return circles if ok else None


def segment_iris(blurred_image: np.ndarray) -> Optional[dict]:
    """Detect pupil and iris boundaries with the production configuration."""
    return segment_iris_configurable(blurred_image, DEFAULT_SEGMENTATION_CONFIG)


def draw_segmentation_overlay(image: np.ndarray, circles: dict) -> np.ndarray:
    """Return a BGR image with pupil and iris circles overlaid for QC."""
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    cx, cy = circles["center"]
    cv2.circle(overlay, (int(cx), int(cy)), int(round(circles["r_iris"])),
               (0, 255, 0), 2)
    cv2.circle(overlay, (int(cx), int(cy)), int(round(circles["r_pupil"])),
               (0, 0, 255), 2)
    cv2.circle(overlay, (int(cx), int(cy)), 3, (255, 0, 0), -1)
    return overlay


def normalize_iris(
    image: np.ndarray,
    pupil_circle: dict,
    iris_circle: dict,
    width: int = 512,
    height: int = 64,
) -> np.ndarray:
    """Unwrap the annular iris region into a rectangular strip.

    Implements Daugman's Rubber-Sheet Model: maps each point in the annular
    region between the pupil and iris boundaries to a point on a 2-D polar
    coordinate grid (rho, theta), producing a fixed-size rectangular image.

    Mapping formula:
        x(rho, theta) = x_pupil(theta) + rho * (x_iris(theta) - x_pupil(theta))
        y(rho, theta) = y_pupil(theta) + rho * (y_iris(theta) - y_pupil(theta))

    where rho ∈ [0, 1] indexes rows (radial direction) and
    theta ∈ [0, 2π) indexes columns (angular direction).

    Parameters
    ----------
    image : np.ndarray
        Denoised grayscale image.
    pupil_circle : dict
        Dict with keys "center" (cx, cy) and "r_pupil".
    iris_circle : dict
        Dict with keys "center" (cx, cy) and "r_iris".
        In practice both dicts come from segment_iris and share the same centre.
    width : int
        Number of angular samples (columns). Default 512.
    height : int
        Number of radial samples (rows). Default 64.

    Returns
    -------
    np.ndarray
        Normalised iris strip of shape (height, width), dtype uint8.
    """
    cx, cy = pupil_circle["center"]
    r_pupil = pupil_circle["r_pupil"]
    r_iris = iris_circle["r_iris"]

    # Build polar coordinate meshgrid
    theta = np.linspace(0, 2 * np.pi, width, endpoint=False)   # (width,)
    rho   = np.linspace(0, 1, height, endpoint=False)           # (height,)
    theta_grid, rho_grid = np.meshgrid(theta, rho)              # (height, width)

    # Boundary points on pupil and iris circumferences
    x_pupil = cx + r_pupil * np.cos(theta_grid)
    y_pupil = cy + r_pupil * np.sin(theta_grid)
    x_iris  = cx + r_iris  * np.cos(theta_grid)
    y_iris  = cy + r_iris  * np.sin(theta_grid)

    # Interpolated Cartesian coordinates for each (rho, theta)
    map_x = (x_pupil + rho_grid * (x_iris - x_pupil)).astype(np.float32)
    map_y = (y_pupil + rho_grid * (y_iris - y_pupil)).astype(np.float32)

    normalized = cv2.remap(
        image,
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return normalized  # (height, width) uint8


def scale_pixels(
    normalized_image: np.ndarray,
    target_shape: tuple = (128, 128),
) -> np.ndarray:
    """Resize the normalised iris strip and convert to a CNN-ready tensor.

    Steps:
      1. Resize to target_shape using bilinear interpolation.
      2. Cast to float32 and divide by 255 → [0.0, 1.0].
      3. Expand the channel dimension → (H, W, 1).

    Parameters
    ----------
    normalized_image : np.ndarray
        Output of normalize_iris, shape (64, 512) uint8.
    target_shape : tuple of int
        (width, height) for cv2.resize. Default (128, 128).

    Returns
    -------
    np.ndarray
        Float32 array of shape (128, 128, 1) with values in [0.0, 1.0].
    """
    resized = cv2.resize(normalized_image, target_shape, interpolation=cv2.INTER_LINEAR)
    scaled  = resized.astype(np.float32) / 255.0
    return np.expand_dims(scaled, axis=-1)
