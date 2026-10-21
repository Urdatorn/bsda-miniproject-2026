"""One look for every figure: the ICML 2025 two-column style.

ICML sets 10 pt Times on a 6.75 in text width with 3.25 in columns, so a
figure here is drawn at one of those two widths in Times New Roman at
7-9 pt, and drops into the paper without rescaling.  The four Times faces
ship in fonts/times.zip (they carry the full polytonic Greek block); apply()
unpacks them next to the zip and registers them with matplotlib, so no
system font is needed.  Should the zip be missing, Libertinus Serif from the
TeX tree stands in, the only other Times-like face at hand with polytonic
Greek (Nimbus Roman, STIX and TeX Gyre Termes do not have it); matplotlib
falls back glyph by glyph along the family list.

  from figure_style import apply, TEXT_WIDTH, COLUMN_WIDTH, font_family
"""
import glob, os, zipfile

import matplotlib
from matplotlib import font_manager

TEXT_WIDTH = 6.75        # inches, the full text width of ICML
COLUMN_WIDTH = 3.25      # one column
FAMILIES = ["Times New Roman", "Libertinus Serif", "DejaVu Serif"]
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
TEX_TREE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".tex", ".TinyTeX")


def register_times():
    """unpack fonts/times.zip (once) and add its four faces to matplotlib"""
    archive = os.path.join(FONT_DIR, "times.zip")
    if not os.path.exists(archive):
        return
    target = os.path.join(FONT_DIR, "times")
    if not glob.glob(os.path.join(target, "*.ttf")):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    for path in glob.glob(os.path.join(target, "**", "*.ttf"), recursive=True):
        font_manager.fontManager.addfont(path)


def register_libertinus():
    """add the Libertinus faces of the TeX tree to matplotlib, if the tree is there"""
    for path in glob.glob(os.path.join(TEX_TREE, "texmf-dist", "fonts", "opentype", "public", "libertinus-fonts", "LibertinusSerif-*.otf")):
        font_manager.fontManager.addfont(path)


def font_family():
    """the first family of FAMILIES that is installed, for libraries that take one name"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for family in FAMILIES:
        if family in installed:
            return family
    return FAMILIES[-1]


# The Okabe-Ito palette: eight colours that colour-blind readers can tell apart; every figure draws on it
OKABE_ITO = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442", "#000000"]


def apply():
    matplotlib.use("Agg")
    register_times()
    register_libertinus()
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": FAMILIES, "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.titlesize": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "savefig.dpi": 300, "figure.dpi": 100,
        "pdf.fonttype": 42, "axes.unicode_minus": False,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO)})     # colour-blind safe default colours
