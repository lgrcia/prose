import argparse
import sys

import matplotlib.pyplot as plt

from prose import FITSImage, FitsManager, Sequence, blocks


def show(args):
    image = FITSImage(args.file)
    image.show(contrast=args.contrast)
    plt.axis(False)
    plt.tight_layout()
    plt.show(block=True)
