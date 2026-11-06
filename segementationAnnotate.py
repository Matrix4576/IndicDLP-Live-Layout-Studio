import argparse
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import numpy.typing as npt

from preprocessing import downscaledSegmentation

PARENT_DIR = Path(__file__).parent
DATA_DIR = PARENT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect whitespace segmentation candidates on training scans."
    )
    parser.add_argument(
        "--min-gap-height",
        type=int,
        default=12,
        help="Minimum consecutive low-ink rows to mark as a candidate gap (pixels).",
    )
    parser.add_argument(
        "--max-gap-density",
        type=float,
        default=0.03,
        help="Maximum ink fraction for a row to count as low ink (0 to 1).",
    )
    return parser.parse_args()


def grayToBinary(grayIMG: cv.typing.MatLike) -> tuple[float, cv.typing.MatLike]:
    """
    Convert a grayscale image to an inverted binary image for ink analysis.
    """
    if not isinstance(grayIMG, np.ndarray) or grayIMG.dtype != np.uint8:
        raise TypeError("grayIMG must be a NumPy array with dtype uint8")
    if grayIMG.ndim != 2 or grayIMG.size == 0:
        raise ValueError("grayIMG must be a non-empty grayscale image")
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


def candidate_regions(
    width: int, x_cuts: npt.NDArray[np.intp]
) -> list[tuple[int, int]]:
    """Split the page at interior X minima into half-open column regions."""
    if width <= 0:
        raise ValueError("width must be positive")
    cuts = np.unique(x_cuts[(x_cuts > 0) & (x_cuts < width)]).astype(int)
    edges = np.concatenate(([0], cuts, [width]))
    return [(int(left), int(right)) for left, right in zip(edges[:-1], edges[1:])]


def low_ink_row_runs(
    binary_region: npt.NDArray[np.uint8],
    max_gap_density: float,
    min_gap_height: int,
) -> tuple[npt.NDArray[np.float64], list[tuple[int, int]]]:
    """Return a region's horizontal profile and qualifying [top, bottom) gaps."""
    if not isinstance(binary_region, np.ndarray) or binary_region.dtype != np.uint8:
        raise TypeError("binary_region must be a NumPy array with dtype uint8")
    if binary_region.ndim != 2 or binary_region.size == 0:
        raise ValueError("binary_region must be a non-empty grayscale region")
    if not 0.0 <= max_gap_density <= 1.0:
        raise ValueError("max_gap_density must be between 0 and 1")
    if min_gap_height < 1:
        raise ValueError("min_gap_height must be at least 1")

    profile = binary_region.astype(np.float64).mean(axis=1) / 255.0
    is_low_ink = profile <= max_gap_density
    transitions = np.diff(np.pad(is_low_ink.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    runs = [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= min_gap_height
    ]
    return profile, runs


def inspect_images(min_gap_height: int, max_gap_density: float) -> None:
    colors = plt.get_cmap("tab10")
    image_files = sorted(path for path in TRAIN_DIR.iterdir() if path.is_file())

    for image_path in image_files:
        img: cv.typing.MatLike | None = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            continue
        downscaled_sample = downscaledSegmentation(img.astype(np.uint8, copy=False))
        _, binary_image = grayToBinary(downscaled_sample)
        height, width = binary_image.shape
        ink_density_x = binary_image.astype(np.float64).sum(axis=0) / (255.0 * height)
        x_minima, x_threshold = weighted_minima_threshold(
            ink_density_x, weight_exponent=0.85
        )
        regions = candidate_regions(width, x_minima)
        local_profiles = [
            (
                left,
                right,
                *low_ink_row_runs(
                    binary_image[:, left:right], max_gap_density, min_gap_height # type: ignore
                ),
            )
            for left, right in regions
        ]

        overlay_height = min(16, max(8, 12 * height / width))
        image_fig, image_ax = plt.subplots(figsize=(12, overlay_height))
        image_ax.imshow(downscaled_sample, cmap="gray", vmin=0, vmax=255)
        image_ax.set_title("X regions and region-local gap candidates")
        image_ax.set_xlabel("X coordinate")
        image_ax.set_ylabel("Y coordinate")

        chart_fig, (x_ax, y_ax) = plt.subplots(1, 2, figsize=(16, 7))

        x_candidates = find_local_minima(ink_density_x)
        x_ax.plot(ink_density_x, color="steelblue", linewidth=0.8, label="Profile")
        x_ax.scatter(
            x_candidates,
            ink_density_x[x_candidates],
            color="gray",
            s=13,
            label="Local minima",
            zorder=3,
        )
        x_ax.set_title(
            "Vertical Projection Profile (X)"
            if x_threshold is None
            else f"Vertical Projection Profile (X), threshold={x_threshold:.3f}"
        )
        x_ax.set_xlabel("X coordinate")
        x_ax.set_ylabel("Ink density")
        x_ax.grid(True)
        x_ax.legend(loc="upper right", fontsize="small")
        if x_threshold is not None:
            x_ax.axhline(x_threshold, color="black", linestyle=":", linewidth=1)

        y_ax.set_title(
            f"Region-local horizontal profiles (minimum gap {min_gap_height}px)"
        )
        y_ax.set_xlabel("Ink density")
        y_ax.set_ylabel("Y coordinate")
        y_ax.set_xlim(0, 1)
        y_ax.set_ylim(height - 1, 0)
        y_ax.grid(True)

        for idx, (left, right, profile, gaps) in enumerate(local_profiles):
            color = colors(idx % 10)
            label = f"Region {idx + 1} ({left}:{right})"
            y_ax.plot(profile, np.arange(height), color=color, linewidth=0.9, label=label)
            for top, bottom in gaps:
                y_ax.axhspan(top, bottom, color=color, alpha=0.14)
                image_ax.add_patch(
                    Rectangle(
                        (left, top),
                        right - left,
                        bottom - top,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.22,
                        linewidth=1.5,
                    )
                )
                image_ax.plot([left, right - 1], [top, top], color=color, linewidth=1.5)
                image_ax.plot(
                    [left, right - 1], [bottom - 1, bottom - 1], color=color, linewidth=1.5
                )
            image_ax.add_patch(
                Rectangle(
                    (left, 0),
                    max(1, right - left - 1),
                    height - 1,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.0,
                    linestyle="--",
                )
            )

        for x_coord_value in x_minima:
            x_coord = int(x_coord_value)
            image_ax.axvline(x_coord, color="red", linewidth=1.5, linestyle=":")
            x_ax.axvline(x_coord, color="red", linestyle="--", linewidth=1.2)
            x_ax.plot(x_coord, ink_density_x[x_coord], "o", color="red")

        y_ax.axvline(
            max_gap_density,
            color="black",
            linestyle=":",
            linewidth=1,
            label=f"Gap density limit ({max_gap_density:.3f})",
        )
        y_ax.legend(loc="upper right", fontsize="x-small")
        image_fig.suptitle(
            f"{image_path.name} - X cuts: {len(x_minima)}, regions: {len(regions)}"
        )
        image_fig.tight_layout()
        chart_fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    if args.min_gap_height < 1:
        raise SystemExit("--min-gap-height must be at least 1")
    if not 0.0 <= args.max_gap_density <= 1.0:
        raise SystemExit("--max-gap-density must be between 0 and 1")
    inspect_images(args.min_gap_height, args.max_gap_density)
