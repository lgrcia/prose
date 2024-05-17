import numpy as np
import pytest

from prose import Fluxes


def test_copy():
    x = np.random.rand(10)
    f = Fluxes(x, data={"test": 0})
    f2 = f.copy()
    assert f.flux is not f2.flux
    assert f.data is not f2.data
    assert f.data == f2.data


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


def test_auto_diff():
    x = np.random.uniform(0, 10000, size=(3, 2, 10))
    f = Fluxes(x)
    f.target = 1
    diff = f.autodiff()


def test_diff():
    x = np.random.uniform(0, 10000, size=(30, 20, 10))
    comps = np.repeat(np.array([[1, 12, 16]]), repeats=30, axis=0)
    f = Fluxes(x)
    f.target = 1
    diff = f.autodiff()
