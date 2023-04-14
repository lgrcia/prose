def test_readme_example():
    import matplotlib.pyplot as plt

    from prose import Sequence, blocks
    from prose.simulations import example_image

    # getting the example image
    image = example_image()

    sequence = Sequence(
        [
            blocks.PointSourceDetection(),  # stars detection
            blocks.Cutouts(shape=21),  # cutouts extraction
            blocks.MedianEPSF(),  # PSF building
            blocks.Moffat2D(),  # PSF modeling
        ]
    )

    sequence.run(image)

    # plotting
    image.show()  # detected stars

    # effective PSF parameters
    image.epsf.params
