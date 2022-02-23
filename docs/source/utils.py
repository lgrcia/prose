from nbconvert import RSTExporter
from prose.io import get_files
import base64
import os
from os import path
import shutil
from traitlets.config import Config

c = Config()
c.RegexRemovePreprocessor.patterns = ["# hidden"]
rst = RSTExporter(config=c)


def save_image(destination, imstring):
    open(destination, 'wb').write(imstring)


def convert_ipynb(filename, destination):
    body, resources = rst.from_filename(filename)
    basename = path.basename(filename)[:-6]
    destination = path.join(destination, basename)

    if path.exists(destination):
        shutil.rmtree(destination)

    os.mkdir(destination)

    for imname, imstring in resources['outputs'].items():
        save_image(path.join(destination, imname), imstring)

    open(path.join(destination, f"{basename}.rst"), "w").write(body.replace("../_static", "../../_static"))
    
import inspect
from os import path
from prose import blocks, Block
from glob import glob

rst_docs = glob(path.join("blocks", "*.rst"))
rst_docs_names = [f.split("/")[-1][0:-4] for f in rst_docs]

for name, obj in inspect.getmembers(blocks):
    if inspect.isclass(obj):
        if issubclass(obj, Block):
            if name not in rst_docs_names:
                filename = path.join("./blocks/others", "{}.rst".format(name))
                with open(filename, "w") as f:
                    f.write(":orphan:\n")
                    f.write(f"\n\n.. autoclass:: prose.blocks.{name}\n\t:members:")
                    print(name)
