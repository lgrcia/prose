from os import path
from astropy.io import fits
import glob


def phot2dict(filename):
    hdu = fits.open(filename)
    dictionary = {h.name.lower(): h.data for h in hdu}
    dictionary["header"] = hdu[0].header

    return dictionary


def get_files(
        ext,
        folder,
        depth=0,
        return_folders=False,
        single_list_removal=True,
        none_for_empty=False,
):
    """

    Return files of specific extension in the specified folder and sub-folders

    Parameters
    ----------
    folder : str
        Folder to be analyzed
    depth : int
        Number how sub-folder layer to look into.
        0 (default) will look into current folder
        1 will look into current folder and its sub-folders
        2 will look into current folder, its sub-folders and their sub-folders
        ... etc

    Returns
    -------
    list of fits files

    """
    files = []
    for depth in range(depth + 1):
        files += glob.iglob(
            path.join(folder, "*/" * depth + "*{}".format(ext)), recursive=False
        )

    files = [path.abspath(f) for f in files]

    if return_folders:
        folders = [path.dirname(file) for file in files]
        if single_list_removal and len(folders) == 1:
            return folders[0]
        else:
            return folders
    else:
        if single_list_removal and len(files) == 1:
            return files[0]
        elif len(files) == 0 and none_for_empty:
            return None
        else:
            return files


def set_hdu(hdu_list, value):
    key = value.name
    if key in hdu_list:
        hdu_list[key] = value
    else:
        hdu_list.append(value)

