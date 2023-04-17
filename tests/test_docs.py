import pytest
import urllib3

from prose.core.block import is_tested


def test_readme_example():
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


def test_block_tested():
    from prose import blocks

    assert is_tested(blocks.PointSourceDetection)
    assert is_tested("PointSourceDetection")
    assert not is_tested(blocks.WriteTo)
    assert not is_tested("WriteTo")


@pytest.mark.skip(reason="takes too long")
def test_readme_urls():
    import re

    # Regular expression pattern to match URLs (not perfect but good enough)
    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .!@#$%^&*()~+-]*"
    # Open the file and read its contents
    with open("README.md", "r") as file:
        file_contents = file.read()

    # Find all URLs in the file using the regular expression pattern
    urls = re.findall(url_pattern, file_contents)
    # Fix the URLS by deleting everything after the ')'
    urls = [url.split(")")[0] for url in urls]

    # check if they are working
    for url in urls:
        http = urllib3.PoolManager()
        error = http.request("GET", url)
        assert error.status != 404, pytest.fail(f"bad '{url}' [{error.status}]")
