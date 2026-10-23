import os
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from preprocessing import downscaledSegmentation

PARENT_DIR = Path(__file__).parent
DATA_DIR = PARENT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"


def grayToBinary(grayIMG: cv.typing.MatLike) -> tuple[float, cv.typing.MatLike]:
    """
    Convert a grayscale image to an inverted binary image for ink analysis.
    """
    threshold_value = 127
    max_value = 255
    threshold, binary_image = cv.threshold (
        grayIMG, threshold_value, max_value, cv.THRESH_BINARY_INV
    )
    return threshold, binary_image


def find_local_minima(profile: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """
    Return coordinates of interior local minima in a one-dimensional profile.
    """
    if profile.ndim != 1 or profile.size < 3:
        return np.empty(0, dtype=np.intp)

    is_minimum = (profile[1:-1] <= profile[:-2]) & (profile[1:-1] < profile[2:])
    return np.flatnonzero(is_minimum).astype(np.intp) + 1


def weighted_minima_threshold (
    profile: npt.NDArray[np.float64],
    weight_exponent: float = 1.0,
) -> tuple[npt.NDArray[np.intp], float | None]:
    """
    Return minima at/below an inverse-power weighted mean for this profile.
    Lower density values receive larger weights, so deep valleys influence the
    image-specific threshold more. We normalize the weights to sum to the number
    of minima, then divide the weighted sum by that count.
    """
    if not np.isfinite(weight_exponent) or weight_exponent <= 0:
        raise ValueError("weight_exponent must be a positive finite number")

    minima = find_local_minima(profile)
    if minima.size == 0:
        return minima, None

    minima_values = profile[minima].astype(np.float64)
    if not np.isfinite(minima_values).all() or np.any(minima_values < 0):
        raise ValueError("profile minima must be finite and non-negative")

    epsilon = max(float(np.max(minima_values)) * 1e-6, 1e-12)
    weights = 1.0 / np.maximum(minima_values, epsilon) ** weight_exponent
    weights *= minima_values.size / float(np.sum(weights))
    threshold = float(np.sum(weights * minima_values) / minima_values.size)
    selected_minima = minima[minima_values <= threshold]
    return selected_minima, threshold


image_files = sorted(os.listdir(TRAIN_DIR))
colors = plt.get_cmap("tab10")

for image_file in image_files:
    image_path = TRAIN_DIR / image_file
    img: cv.typing.MatLike | None = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        continue

    downscaled_sample = downscaledSegmentation(img.astype(np.uint8, copy=False))
    _, binary_image = grayToBinary(downscaled_sample)
    height, width = binary_image.shape[:2]

    ink_density_x = binary_image.sum(axis=0) / (255 * height)
    ink_density_y = binary_image.sum(axis=1) / (255 * width)
    # A gentler inverse weight on X keeps very low scan-edge minima from
    # pulling the cutoff down as strongly, while still favoring deeper valleys.
    x_minima, x_threshold = weighted_minima_threshold(
        ink_density_x, weight_exponent=0.85
    )
    y_minima, y_threshold = weighted_minima_threshold(ink_density_y)

    annotated_bgr = cv.cvtColor(downscaled_sample, cv.COLOR_GRAY2BGR)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    axes[0].imshow(cv.cvtColor(annotated_bgr, cv.COLOR_BGR2RGB))
    axes[0].set_title("Image with weighted-minimum lines")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")

    x_candidates = find_local_minima(ink_density_x)
    axes[1].plot(ink_density_x, color="steelblue", linewidth=0.8, label="Profile")
    axes[1].scatter(
        x_candidates,
        ink_density_x[x_candidates],
        color="gray",
        s=13,
        label="Local minima",
        zorder=3,
    )
    axes[1].set_title(
        "Vertical Projection Profile (X)"
        if x_threshold is None
        else f"Vertical Projection Profile (X), threshold={x_threshold:.3f}"
    )
    axes[1].set_xlabel("X coordinate")
    axes[1].set_ylabel("Ink density")
    axes[1].grid(True)
    axes[1].legend(loc="upper right", fontsize="small")
    if x_threshold is not None:
        axes[1].axhline(x_threshold, color="black", linestyle=":", linewidth=1)

    y_candidates = find_local_minima(ink_density_y)
    axes[2].plot(ink_density_y, color="steelblue", linewidth=0.8, label="Profile")
    axes[2].scatter(
        y_candidates,
        ink_density_y[y_candidates],
        color="gray",
        s=13,
        label="Local minima",
        zorder=3,
    )
    axes[2].set_title(
        "Horizontal Projection Profile (Y)"
        if y_threshold is None
        else f"Horizontal Projection Profile (Y), threshold={y_threshold:.3f}"
    )
    axes[2].set_xlabel("Y coordinate")
    axes[2].set_ylabel("Ink density")
    axes[2].grid(True)
    axes[2].legend(loc="upper right", fontsize="small")
    if y_threshold is not None:
        axes[2].axhline(y_threshold, color="black", linestyle=":", linewidth=1)

    for idx, x_coord_value in enumerate(x_minima):
        x_coord = int(x_coord_value)
        color = colors(idx % 10)
        bgr_color = tuple(int(channel * 255) for channel in color[2::-1])
        cv.line(annotated_bgr, (x_coord, 0), (x_coord, height - 1), bgr_color, 2)
        axes[1].axvline(x_coord, color=color, linestyle="--", linewidth=1.4)
        axes[1].plot(x_coord, ink_density_x[x_coord], "o", color=color)

    for idx, y_coord_value in enumerate(y_minima):
        y_coord = int(y_coord_value)
        color = colors((idx + 5) % 10)
        bgr_color = tuple(int(channel * 255) for channel in color[2::-1])
        cv.line(annotated_bgr, (0, y_coord), (width - 1, y_coord), bgr_color, 2)
        axes[2].axvline(y_coord, color=color, linestyle="--", linewidth=1.4)
        axes[2].plot(y_coord, ink_density_y[y_coord], "o", color=color)

    axes[0].images[0].set_data(cv.cvtColor(annotated_bgr, cv.COLOR_BGR2RGB))
    plt.suptitle(
        f"{image_file} - selected minima: X={len(x_minima)}, Y={len(y_minima)}"
    )
    plt.tight_layout()
    plt.show()
