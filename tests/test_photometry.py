import numpy as np

from prose import Sequence, blocks, simulations

t = np.linspace(0, 1, 20)
true_y = np.sin(2 * np.pi * t / 0.5) + 1.0

np.random.seed(5)

fluxes = np.array([true_y, *[np.ones(len(t)) for _ in range(20)]])
_coords = np.random.rand(len(fluxes), 2)
shape = (100, 100)
coords = np.array([_coords * shape for _ in range(len(t))])
coords[:, 0, :] = np.array(shape) / 2

images = simulations.simple_images(fluxes, coords, 1.0, shape=shape)


def test_photometry():
    ref = images[0]
    ref = blocks.PointSourceDetection(False, 0, 0)(ref)

    def set_sources(im):
        im.sources = ref.sources.copy()

    photometry = Sequence(
        [
            blocks.Apply(set_sources),
            blocks.AperturePhotometry(radii=[2.0], scale=False),
            blocks.AnnulusBackground(scale=False),
            blocks.GetFluxes(),
        ]
    )

    photometry.run(images)
    fluxes = photometry[-1].fluxes
    fluxes.aperture = 0
    fluxes.target = 14
    assert np.allclose(true_y, fluxes.flux)
