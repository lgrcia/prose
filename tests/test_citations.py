import pytest

from prose import Image
from prose.blocks.background import PhotutilsBackground2D


def test_photutils_background_citation():
    block = PhotutilsBackground2D()
    assert "photutils" in block.citations
