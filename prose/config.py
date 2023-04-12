import glob
import shutil
from os import path
from pathlib import Path

import numpy as np
import requests
import yaml
from yaml import Loader

from .builtins import built_in_telescopes

info = print
package_name = "prose"


class ConfigManager:
    def __init__(self):
        self.config = None

        self.folder_path = Path.home() / f".{package_name}"
        self.folder_path.mkdir(exist_ok=True)

        self.config_file = self.folder_path / "config"

        self.rename_id_files_to_telescopes()
        self.create_builtins_telescopes_files()
        self.check_config_file(load=True)
        self.telescopes_dict = self.build_telescopes_dict()
        self.check_ballet()
        self.logs = []

    def check_config_file(self, load=False):
        if self.config_file.exists():
            with self.config_file.open(mode="r") as file:
                if load:
                    self.config = yaml.load(file.read(), Loader=Loader)
        else:
            info(f"A config file as been created in {self.folder_path}")
            self.config = {"color": "blue"}
            with self.config_file.open(mode="w") as file:
                yaml.dump(self.config, file, default_flow_style=False)

    def save(self):
        self.check_config_file()
        with self.config_file.open(mode="w") as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def get(self, key):
        return self.config.get(key)

    def set(self, key, value):
        self.config[key] = value
        self.save()

    def rename_id_files_to_telescopes(self):
        """For backward compat with 0.9.6 downward"""
        id_files = list(self.folder_path.glob("*.id"))
        if len(id_files) > 0:
            info("Renaming some old .id telescope files")
            for fpath in id_files:
                shutil.move(fpath, str(fpath).replace(".id", ".telescope"))

    def check_builtins_changes(self):
        for name, telescope in built_in_telescopes.items():
            telescope_file_name = path.join(self.folder_path, f"{name}.telescope")
            if path.exists(telescope_file_name):
                with open(telescope_file_name, mode="r") as f:
                    existing_telescope = yaml.load(f, Loader=yaml.FullLoader)
                    if existing_telescope != telescope:
                        info(
                            f"{Path(telescope_file_name).name} differs from builtins (use prose.CONFIG.update_builtins() to update)"
                        )

    def update_builtins(self):
        self.create_builtins_telescopes_files(force=True)

    def build_telescopes_dict(self):
        telescope_files = self.folder_path.glob("*.telescope")

        telescope_dict = {}

        for telescope_file in telescope_files:
            with telescope_file.open(mode="r") as f:
                telescope = yaml.load(f, Loader=yaml.FullLoader)
            telescope_dict[telescope["name"].lower()] = telescope
            if "names" in telescope:
                for name in telescope["names"]:
                    telescope_dict[name.lower()] = telescope

        # telescope_dict.update(built_in_telescopes)

        return telescope_dict

    def create_builtins_telescopes_files(self, force=False):
        for name, telescope in built_in_telescopes.items():
            telescope_file_name = path.join(self.folder_path, f"{name}.telescope")
            if (not path.exists(telescope_file_name) and not force) or force:
                self.save_telescope_file(telescope)

    def save_telescope_file(self, file):
        if isinstance(file, str):
            name = Path(file).stem.lower()
            shutil.copyfile(file, self.folder_path / f"{name}.telescope")
            info("Telescope '{}' saved".format(name))
        elif isinstance(file, dict):
            name = file["name"].lower()
            telescope_file_path = self.folder_path / f"{name}.telescope"
            yaml.dump(file, telescope_file_path.open(mode="w"))
            info("Telescope '{}' saved".format(name))
        else:
            raise AssertionError("input type not understood")
        self.telescopes_dict = self.build_telescopes_dict()

    def match_telescope_name(self, name):
        available_telescopes_names = list(self.telescopes_dict.keys())
        has_telescope = np.flatnonzero(
            [t.lower() == name.lower() for t in available_telescopes_names]
        )
        if len(has_telescope) > 0:
            i = np.argmax(
                [
                    len(name)
                    for name in np.array(available_telescopes_names)[has_telescope]
                ]
            )
            return self.telescopes_dict[available_telescopes_names[has_telescope[i]]]
        else:
            return None

    def check_ballet(self):
        model_path = self.folder_path / "centroid.h5"

        if not model_path.exists():
            print("downloading ballet model (~30Mb)")
            model = requests.get(
                "https://github.com/lgrcia/ballet/raw/master/models/centroid.h5"
            ).content
            model_path.open(mode="wb").write(model)
