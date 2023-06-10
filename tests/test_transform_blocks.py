import numpy as np
import pytest
from skimage import transform

from prose import Image, blocks
from prose.core import Sources


def test_transform_block(n=100):
    np.random.seed(n)
    original_transform = transform.AffineTransform(
        rotation=np.pi / 3, translation=(100, 100), scale=14.598
    )

    original_coords = np.random.rand(n, 2) * 1000
    original_image = Image(_sources=Sources(original_coords))

    transformed_image = Image(
        _sources=Sources(original_transform.inverse(original_coords))
    )

    block_transform = blocks.geometry.ComputeTransform(original_image)
    computed_transform = block_transform(transformed_image).transform

    assert computed_transform.rotation == pytest.approx(original_transform.rotation)
    assert computed_transform.translation == pytest.approx(
        original_transform.translation
    )
    assert computed_transform.scale == pytest.approx(original_transform.scale)
