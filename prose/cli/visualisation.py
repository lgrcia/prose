from pathlib import Path

import click
import matplotlib.pyplot as plt

from prose import FITSImage, FitsManager, Sequence, blocks


@click.command(name="show", help="show FITS image")
@click.argument("file")
@click.option(
    "-c",
    "--contrast",
    type=float,
    help="contrast of the image (zscale is applied)",
    default=0.1,
)
@click.option(
    "-f",
    "--frame",
    is_flag=True,
    help="whether to show sky coordinates frame",
)
def show(file, contrast, frame):
    image = FITSImage(file)
    if frame and not image.plate_solved:
        print("Image is not plate solved, cannot show frame")
        frame = False
    image.show(contrast=contrast, frame=frame)
    if not frame:
        plt.axis(False)
    plt.tight_layout()
    plt.show(block=True)


@click.command(name="video", help="make a video of FITS images")
@click.argument("folder")
@click.option(
    "-d",
    "--depth",
    type=int,
    help="subfolder parsing depth",
    default=10,
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="output video file",
    default="video.mp4",
)
@click.option(
    "-t",
    "--type",
    type=str,
    help="type of FITS files to use",
    default=None,
)
@click.option(
    "-f",
    "--fps",
    type=int,
    help="frames per second",
    default=10,
)
@click.option(
    "-c",
    "--compression",
    type=int,
    help="compression parameter for the video block",
    default=None,
)
@click.option(
    "-w",
    "--width",
    type=int,
    help="width of the video in pixel (if resizing required), aspect ratio is kept",
    default=None,
)
def video(folder, depth, output, type, fps, compression, width):
    fm = FitsManager(folder, depth=depth)
    images = fm.files(type="*" if type is None else type, path=True).path.values

    if output is None:
        output = Path(folder) / "video.mp4"

    else:
        output = Path(output)

    video_sequence = Sequence(
        [
            blocks.Video(
                output,
                fps=fps,
                compression=compression,
                width=width,
            )
        ],
        name="Making video",
    )

    video_sequence.run(images)
    print("Video saved in", output)
