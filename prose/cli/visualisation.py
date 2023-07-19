from pathlib import Path

import matplotlib.pyplot as plt

from prose import FITSImage, FitsManager, Sequence, blocks


def show(args):
    image = FITSImage(args.file)
    if args.f and not image.plate_solved:
        print("Image is not plate solved, cannot show frame")
        args.f = False
    image.show(contrast=args.contrast, frame=args.f)
    if not args.f:
        plt.axis(False)
    plt.tight_layout()
    plt.show(block=True)


def add_show_parser(subparsers):
    show_parser = subparsers.add_parser(name="show", description="show FITS image")
    show_parser.add_argument("file", type=str, help="file to show", default=None)
    show_parser.add_argument(
        "-c",
        "--contrast",
        type=float,
        help="contrast of the image (zscale is applied)",
        default=0.1,
    )
    show_parser.add_argument(
        "-f",
        action="store_true",
        help="whether to show sky coordinates frame",
    )
    show_parser.set_defaults(func=show)


def video(args):
    fm = FitsManager(args.folder, depth=args.depth)
    images = fm.files(
        type="*" if args.type is None else args.type, path=True
    ).path.values

    if args.output is None:
        output = Path(args.folder) / "video.mp4"

    else:
        output = Path(args.output)

    video_sequence = Sequence(
        [
            blocks.Video(
                output,
                fps=args.fps,
                compression=args.compression,
                width=args.width,
            )
        ],
        name="Making video",
    )

    video_sequence.run(images)
    print("Video saved in", output)


def add_video_parser(subparsers):
    video_parser = subparsers.add_parser(
        name="video", description="make a video of FITS images"
    )
    video_parser.add_argument(
        "folder", type=str, help="folder containing the FITS", default=None
    )
    video_parser.add_argument(
        "-d", "--depth", type=int, help="subfolder parsing depth", default=10
    )
    video_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output video file",
        default="video.mp4",
    )
    video_parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="type of FITS files to use",
        default=None,
    )
    video_parser.add_argument(
        "-f",
        "--fps",
        type=int,
        help="frames per second",
        default=10,
    )
    video_parser.add_argument(
        "-c",
        "--compression",
        type=int,
        help="compression parameter for the video block",
        default=None,
    )
    video_parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="width of the video in pixel (if resizing required), aspect ratio is kept",
        default=None,
    )

    video_parser.set_defaults(func=video)
