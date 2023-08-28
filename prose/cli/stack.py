import argparse
from pathlib import Path

import click

from prose import FITSImage, FitsManager, Sequence, blocks
from prose.core.sequence import SequenceParallel


@click.command(name="stack", help="stack FITS images")
@click.argument("folder")
@click.option(
    "-d",
    "--depth",
    type=int,
    help="subfolder parsing depth",
    default=10,
)
@click.option(
    "-n",
    "--n",
    type=int,
    help="number of stars used for alignment",
    default=30,
)
@click.option(
    "--method",
    type=click.Choice(["mean", "selective"]),
    help="alignment method. 'mean' applies a mean to all images, 'selective' \
        applies a median to the -n smallest-FWHM images",
    default="selective",
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="output file name",
    default="stack.fits",
)
def stack(folder, depth, n, method, output):
    folder = Path(folder)

    fm = FitsManager(folder, depth=depth)
    calibrated_nights = fm.observations(type="calibrated")
    files = fm.files(int(calibrated_nights.index[0]), path=True).path.values

    # reference
    ref = FITSImage(files[len(files) // 2])

    # calibration
    psf_sequence = Sequence(
        [
            blocks.PointSourceDetection(n=n),  # stars detection
            blocks.Cutouts(21),  # stars cutouts
            blocks.MedianEPSF(),  # building EPSF
            blocks.psf.Moffat2D(),  # modeling EPSF
        ]
    )

    psf_sequence.run(ref, show_progress=False)

    stack_block = (
        blocks.SelectiveStack(n=50)
        if method == "selective"
        else blocks.MeanStack(reference=ref)
    )

    stacking_sequence = SequenceParallel(
        [
            blocks.PointSourceDetection(n=n),  # stars detection
            blocks.Cutouts(21),  # stars cutouts
            blocks.MedianEPSF(),  # building EPSF
            blocks.psf.Moffat2D(ref),  # modeling EPSF
            blocks.ComputeTransformTwirl(ref),
            blocks.TransformData(inverse=True),
        ],
        [
            stack_block,
        ],
        name="Stacking",
    )

    stacking_sequence.run(files)

    stack = stack_block.stack
    stack.header = ref.header.copy()
    stack.header[ref.telescope.keyword_image_type] = "stack"
    stack.writeto(folder / "stack.fits")

    print("Stack saved in", folder / "stack.fits")
