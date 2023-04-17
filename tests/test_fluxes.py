import numpy as np
import pytest

from prose import Fluxes


def test_1d():
    x = np.random.rand(10)
    f = Fluxes(x)
    f.flux

    # with errors
    with pytest.raises(AssertionError) as excinfo:
        f.error
    assert "errors not provided" in str(excinfo.value)

    f = Fluxes(fluxes=x, errors=x)
    f.error


def test_2d():
    x = np.random.rand(2, 10)
    f = Fluxes(x)
    with pytest.raises(AssertionError) as excinfo:
        f.flux
        f.error
    assert "target must be set" in str(excinfo.value)
    f.target = 0
    f.flux

    # with errors
    with pytest.raises(AssertionError) as excinfo:
        f.error
    assert "errors not provided" in str(excinfo.value)

    f = Fluxes(fluxes=x, errors=x, target=0)
    f.error


def test_3d():
    x = np.random.rand(3, 2, 10)
    f = Fluxes(x)
    with pytest.raises(AssertionError) as excinfo:
        f.flux
        f.error
    assert "target must be set" in str(excinfo.value)
    f.target = 0
    with pytest.raises(AssertionError) as excinfo:
        f.flux
        f.error
    assert "aperture must be set" in str(excinfo.value)
    f.aperture = 0
    f.flux

    # with errors
    with pytest.raises(AssertionError) as excinfo:
        f.error
    assert "errors not provided" in str(excinfo.value)

    f = Fluxes(fluxes=x, errors=x, target=0, aperture=0)
    f.error
