import argparse
from pathlib import Path

from prose import FITSImage, FitsManager, Sequence, blocks


def solve(args):
    if args.output is None:
        output = Path(args.file_or_folder)
    else:
        output = Path(args.output)

    if Path(args.file_or_folder).is_file():
        solve_sequence = Sequence(
            [blocks.PointSourceDetection(n=30), blocks.PlateSolve()]
        )
        image = FITSImage(args.file_or_folder)
        solve_sequence.run(image)
        image.writeto(output)
    else:
        fm = FitsManager(args.file_or_folder, depth=args.depth)
        images = fm.files(
            type="*" if args.type is None else args.type, path=True
        ).path.values

        if args.reference is None:
            reference = images[int(len(images) // 2)]
        else:
            reference = args.reference
        reference = FITSImage(reference)

        Sequence(
            [
                blocks.PointSourceDetection(n=30),
                blocks.PlateSolve(),
                blocks.GaiaCatalog(limit=30),
            ]
        ).run(reference, show_progress=False)

        solve_sequence = Sequence(
            [
                blocks.PointSourceDetection(n=30),
                blocks.ComputeTransformTwirl(reference_image=reference),
                blocks.AlignReferenceSources(reference=reference),
                blocks.AlignReferenceWCS(reference=reference),
                blocks.WriteTo(output, overwrite=True),
            ],
            name="Plate solving",
        )
        solve_sequence.run(images)


def add_solve_parser(subparsers):
    solve_parser = subparsers.add_parser(
        name="solve", description="Plate solve one or several FITS images"
    )
    solve_parser.add_argument(
        "file_or_folder", type=str, help="file/folder to plate solve", default=None
    )
    solve_parser.add_argument(
        "-d", "--depth", type=int, help="subfolder parsing depth", default=10
    )
    solve_parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="type of FITS files to plate solve",
        default=None,
    )
    solve_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output file and/or folder. If leave to default files are overwritten",
        default="input",
    )
    solve_parser.add_argument(
        "-r", "--reference", type=str, help="reference image", default=None
    )

    solve_parser.set_defaults(func=solve)
