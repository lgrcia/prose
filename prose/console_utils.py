import os
import shlex
import struct
import platform
import subprocess
from colorama import Fore
from . import CONFIG
from tqdm import tqdm as _tqdm

style = {
        "spinner_color": {
            "bright_green": "green",
            "bright_blue": "blue",
            "yellow": "yellow",
            "blue": "blue",
            "red": "red",
            "white": "white",
        },
        "fore_color": {
            "bright_green": Fore.LIGHTGREEN_EX,
            "bright_blue": Fore.LIGHTBLUE_EX,
            "yellow": Fore.YELLOW,
            "blue": Fore.LIGHTBLUE_EX,
            "red": Fore.RED,
            "white": Fore.WHITE,
        },
        "info_label": {}
}

SPINNER_COLOR = style["spinner_color"][CONFIG.config["color"]]
FORE_COLOR = style["fore_color"][CONFIG.config["color"]]

RUN_LABEL = "{}RUN{}".format(FORE_COLOR, Fore.RESET)
INFO_LABEL = "{}INFO{}".format(FORE_COLOR, Fore.RESET)
WARNING_LABEL = "{}WARNING{}".format(style["fore_color"]["yellow"], Fore.RESET)

TQDM_BAR_FORMAT = "%s {l_bar}%s{bar}%s{r_bar}" % (
    RUN_LABEL, FORE_COLOR, Fore.RESET
)

def tqdm(x, desc="run", unit="images"):
    return _tqdm(x, desc=desc, unit=unit, ncols=80, bar_format=TQDM_BAR_FORMAT)

def get_terminal_size():
    """ getTerminalSize()
     - get width and height of console
     - works on linux,os x,windows,cygwin(windows)
     originally retrieved from:
     http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
    """
    current_os = platform.system()
    tuple_xy = None
    if current_os == "Windows":
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os in ["Linux", "Darwin"] or current_os.startswith("CYGWIN"):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        print
        "default"
        tuple_xy = (80, 25)  # default value

    return tuple_xy


def _get_terminal_size_windows():
    try:
        from ctypes import windll, create_string_buffer

        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12
        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (
                bufx,
                bufy,
                curx,
                cury,
                wattr,
                left,
                top,
                right,
                bottom,
                maxx,
                maxy,
            ) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
            return sizex, sizey
    except:
        pass


def _get_terminal_size_tput():
    # get terminal width
    # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width-height-of-a-terminal-window
    try:
        cols = int(subprocess.check_call(shlex.split("tput cols")))
        rows = int(subprocess.check_call(shlex.split("tput lines")))
        return (cols, rows)
    except:
        pass


def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios

            cr = struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))
            return cr
        except:
            pass

    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (os.environ["LINES"], os.environ["COLUMNS"])
        except:
            return None

    return int(cr[1]), int(cr[0])


def info(s):
    print(f"{INFO_LABEL} {s}")


def warning(s):
    print(f"{WARNING_LABEL} {s}")