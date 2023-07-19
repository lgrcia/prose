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
