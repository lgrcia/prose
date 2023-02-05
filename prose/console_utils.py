import os
import shlex
import struct
import platform
import subprocess
import warnings
from . import CONFIG
from datetime import datetime
from tqdm import TqdmExperimentalWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)
    from tqdm.autonotebook import tqdm

def color(s, i):
    return f"\u001b[38;5;{i}m{s}\x1b[0m"

TQDM_BAR_FORMAT = "%s {l_bar}%s{bar}%s{r_bar}" % (
    color("RUN", 12), "\u001b[38;5;12m", "\x1b[0m"
)

def progress(show, **kwargs):
    if show:
        return lambda x: tqdm(x, **kwargs)
    else:
        return lambda x: x

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



def _log(type, s):
    date = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    for file in CONFIG.logs:
        open(file, "a").write(f"{date} {type} {s}\n")

def log(s):
    print(s)
    for file in CONFIG.logs:
        open(file, "a").write(f"{s}\n")

def info(s):
    print(f"{color('INFO', 12)} {s}")
    _log('INFO', s)

def warning(s):
    print(f"{color('WARNING', 3)} {s}")
    _log('WARNING', s)

def error(s):
    print(f"{color('ERROR', 1)} {s}")
    _log('ERROR', s)