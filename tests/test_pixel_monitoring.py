import numpy as np
import pytest
from PIL import Image

from pet_breed_mlops.pixel_monitoring import (
    calculate_pixel_statistics,
    statistics_to_dict,
)


def test_calculates_statistics_for_constant_rgb_image() -> None:
    image = Image.new("RGB", (4, 4), color=(255, 128, 0))

    stats = calculate_pixel_statistics(image)

    assert stats.mean_rgb == pytest.approx((1.0, 128 / 255, 0.0))
    assert stats.std_rgb == pytest.approx((0.0, 0.0, 0.0))
    assert stats.mean_brightness == pytest.approx(
        0.2126 + 0.7152 * (128 / 255)
    )


def test_detects_pixel_variation() -> None:
    pixels = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[0, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels)

    stats = calculate_pixel_statistics(image)

    assert stats.mean_rgb == pytest.approx((0.5, 0.5, 0.5))
    assert stats.std_rgb == pytest.approx((0.5, 0.5, 0.5))
    assert stats.mean_brightness == pytest.approx(0.5)


def test_supports_grayscale_images() -> None:
    image = Image.new("L", (3, 3), color=128)

    stats = calculate_pixel_statistics(image)

    assert stats.mean_rgb == pytest.approx((128 / 255,) * 3)
    assert stats.std_rgb == pytest.approx((0.0, 0.0, 0.0))


def test_statistics_to_dict_returns_json_compatible_values() -> None:
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))

    result = statistics_to_dict(calculate_pixel_statistics(image))

    assert set(result) == {
        "mean_rgb",
        "std_rgb",
        "mean_brightness",
        "edge_strength",
        "jpeg_blockiness",
    }
    assert isinstance(result["mean_rgb"], tuple)
    assert all(isinstance(value, float) for value in result["mean_rgb"])

from pet_breed_mlops.pixel_monitoring import (
    PixelStatistics,
    compare_pixel_statistics,
)


@pytest.fixture
def pixel_baseline() -> dict[str, object]:
    return {
        "statistics": {
            "edge_strength": {"mean": 0.03, "std": 0.01},
            "jpeg_blockiness": {"mean": 0.02, "std": 0.01},
            "mean_rgb": {
                "mean": [0.5, 0.4, 0.3],
                "std": [0.1, 0.1, 0.1],
            },
            "std_rgb": {
                "mean": [0.2, 0.2, 0.2],
                "std": [0.05, 0.05, 0.05],
            },
            "mean_brightness": {"mean": 0.4, "std": 0.1},
        }
    }


def test_pixel_comparison_is_zero_for_matching_statistics(
    pixel_baseline: dict[str, object],
) -> None:
    current = [
        PixelStatistics(
            mean_rgb=(0.5, 0.4, 0.3),
            std_rgb=(0.2, 0.2, 0.2),
            mean_brightness=0.4,
            edge_strength=0.03,
            jpeg_blockiness=0.02,

        )
    ]

    result = compare_pixel_statistics(pixel_baseline, current)

    assert result["image_count"] == 1
    assert result["score"] == pytest.approx(0.0)


def test_pixel_comparison_increases_for_shifted_statistics(
    pixel_baseline: dict[str, object],
) -> None:
    current = [
        PixelStatistics(
            mean_rgb=(0.8, 0.4, 0.3),
            std_rgb=(0.2, 0.2, 0.2),
            mean_brightness=0.4,
            edge_strength=0.0,
            jpeg_blockiness=0.02,
        )
    ]

    result = compare_pixel_statistics(pixel_baseline, current)

    assert result["score"] > 0.0
    assert result["features"]["mean_rgb"]["standardized_difference"][0] == pytest.approx(3.0)


def test_pixel_comparison_rejects_empty_batch(
    pixel_baseline: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compare_pixel_statistics(pixel_baseline, [])


def test_pixel_comparison_rejects_non_positive_epsilon(
    pixel_baseline: dict[str, object],
) -> None:
    current = [
        PixelStatistics(
            mean_rgb=(0.5, 0.4, 0.3),
            std_rgb=(0.2, 0.2, 0.2),
            mean_brightness=0.4,
            edge_strength=0.0,
            jpeg_blockiness=0.02,
        )
    ]

    with pytest.raises(ValueError, match="epsilon"):
        compare_pixel_statistics(pixel_baseline, current, epsilon=0)

def test_pixel_statistics_include_edge_strength() -> None:
    smooth_image = Image.new("RGB", (16, 16), color=(128, 128, 128))

    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[:, 8:, :] = 255
    edge_image = Image.fromarray(pixels)

    smooth_stats = calculate_pixel_statistics(smooth_image)
    edge_stats = calculate_pixel_statistics(edge_image)

    assert hasattr(smooth_stats, "edge_strength")
    assert edge_stats.edge_strength > smooth_stats.edge_strength

def test_pixel_statistics_include_jpeg_blockiness() -> None:
    smooth_image = Image.new("RGB", (32, 32), color=(128, 128, 128))

    pixels = np.full((32, 32, 3), 128, dtype=np.uint8)
    pixels[:, 8::8, :] = 255
    block_image = Image.fromarray(pixels)

    smooth_stats = calculate_pixel_statistics(smooth_image)
    block_stats = calculate_pixel_statistics(block_image)

    assert hasattr(smooth_stats, "jpeg_blockiness")
    assert block_stats.jpeg_blockiness > smooth_stats.jpeg_blockiness