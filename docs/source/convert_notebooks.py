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

    open(path.join(destination, f"{basename}.rst"), "w").write(
        body.replace("../_static", "../../_static")
        )
    
    
for file in get_files("ipynb", "./notebooks"):
    convert_ipynb(file, "./tutorials")