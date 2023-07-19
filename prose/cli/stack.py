import argparse
from pathlib import Path

from prose import FITSImage, FitsManager, Sequence, blocks
from prose.core.sequence import SequenceParallel


def stack(args):
    folder = Path(args.folder)

    fm = FitsManager(folder, depth=args.depth)
    calibrated_nights = fm.observations(type="calibrated")
    files = fm.files(int(calibrated_nights.index[0]), path=True).path.values

    # reference
    ref = FITSImage(files[len(files) // 2])

    # calibration
    psf_sequence = Sequence(
        [
            blocks.PointSourceDetection(n=args.n),  # stars detection
            blocks.Cutouts(21),  # stars cutouts
            blocks.MedianEPSF(),  # building EPSF
            blocks.psf.Moffat2D(),  # modeling EPSF
        ]
    )

    psf_sequence.run(ref, show_progress=False)

    stack_block = (
        blocks.SelectiveStack(n=50)
        if args.method == "selective"
        else blocks.MeanStack(reference=ref)
    )

    stacking_sequence = SequenceParallel(
        [
            blocks.PointSourceDetection(n=args.n),  # stars detection
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


def add_stack_parser(subparsers):
    stack_parser = subparsers.add_parser(name="stack", description="stack FITS files")
    stack_parser.add_argument("folder", type=str, help="folder to parse", default=None)
    stack_parser.add_argument(
        "-d", "--depth", type=int, help="subfolder parsing depth", default=10
    )
    stack_parser.add_argument(
        "-n", "--n", type=int, help="number of stars used for alignment", default=30
    )
    stack_parser.add_argument(
        "--method",
        choices=["mean", "selective"],
        help="alignment method. 'mean' applies a mean to all images, 'selective' \
            applies a median to the -n smallest-FWHM images",
        default="selective",
    )
    stack_parser.add_argument(
        "-o", "--output", type=str, help="output file name", default="stack.fits"
    )
    stack_parser.set_defaults(func=stack)
