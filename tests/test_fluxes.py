from prose import Fluxes
import numpy as np
import pytest


def test_1d():
    f = Fluxes(np.random.rand(10))
    f.flux


def test_2d():
    f = Fluxes(np.random.rand(2, 10))
    with pytest.raises(AssertionError) as excinfo:
        f.flux
    assert "target must be set" in str(excinfo.value)
    f.target = 0
    f.flux


def test_3d():
    f = Fluxes(np.random.rand(3, 2, 10))
    with pytest.raises(AssertionError) as excinfo:
        f.flux
    assert "target must be set" in str(excinfo.value)
    f.target = 0
    with pytest.raises(AssertionError) as excinfo:
        f.flux
    assert "aperture must be set" in str(excinfo.value)
    f.aperture = 0
    f.flux

