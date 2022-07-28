import os
from os import path
import shutil
import jinja2
from .. import viz
from pathlib import Path


template_folder = path.abspath(path.join(path.dirname(__file__), "..", "..", "latex"))


class LatexTemplate:
    def __init__(self, template_name=None, style="paper"):
        self.template_name = template_name
        self._style = style
        self.template = None
        self.dpi=150
        if template_name is not None:
            self.load_template()

        # to be set
        self.destination = None
        self.report_name = None
        self.figure_destination = None
        self.tex_destination = None

    def style(self):
        if self._style == "paper":
            viz.paper_style()
        elif self._style == "bokeh":
            viz.bokeh_style()

    @property
    def clean_name(self):
        return self.template_name.replace(".tex", "")

    def load_template(self):
        latex_jinja_env = jinja2.Environment(
            block_start_string='\BLOCK{',
            block_end_string='}',
            variable_start_string='\VAR{',
            variable_end_string='}',
            comment_start_string='\#{',
            comment_end_string='}',
            line_statement_prefix='%%',
            line_comment_prefix='%#',
            trim_blocks=True,
            autoescape=False,
            loader=jinja2.FileSystemLoader(template_folder)
        )
        self.template = latex_jinja_env.get_template(self.template_name)

    def make_report_folder(self, destination, figures=True):
        destination = Path(destination)
        destination.mkdir(exist_ok=True)
        self.destination = destination
        self.report_name = destination.stem
        if figures:
            self.figure_destination =  destination / "figures"
            self.figure_destination.mkdir(exist_ok=True)
        self.tex_destination = destination / f"{self.report_name}.tex"


class Report(LatexTemplate):

    def __init__(self, reports, template_name="report.tex"):
        LatexTemplate.__init__(self, template_name)
        self.reports = reports
        self.paths = None

    def compile(self, destination=None, verbose=False):
        cwd = os.getcwd()
        os.chdir(self.destination)
        os.system(f"pdflatex {self.report_name}{'' if verbose else ' >/dev/null 2>&1'}")
        os.chdir(cwd)

        if destination is not None:
            shutil.copy(str(self.destination / self.report_name) + ".pdf", destination)
            shutil.rmtree(self.destination)

    def make(self, destination):
        self.make_report_folder(destination, figures=False)
        shutil.copyfile(path.join(template_folder, "prose-report.cls"), path.join(destination, "prose-report.cls"))
        tex_destination = path.join(self.destination, f"{self.report_name}.tex")

        self.paths = []
        for report in self.reports:
            print(f"making {report.clean_name} ...")
            report_path = path.join(destination, report.clean_name)
            report.make(report_path)
            self.paths.append(path.join(report.clean_name, report.clean_name))

        open(tex_destination, "w").write(self.template.render(
            paths=self.paths
        ))


def copy_figures(folder, prefix, destination):
    figures = list(Path(folder).glob("**/*.png"))
    texts = list(Path(folder).glob("**/*.txt"))
    pdfs = list(Path(folder).glob("**/*.pdf"))
    new_folder = Path(destination)
    new_folder.mkdir(exist_ok=True)
    for fig in figures:
        if ".ipynb_checkpoints" not in str(fig):
            shutil.copy(fig, new_folder / (prefix + "_" + fig.name))
    for txt in texts:
        if ".ipynb_checkpoints" not in str(txt):
            shutil.copy(txt, new_folder / (prefix + "_" + txt.name))
    for pdf in pdfs:
        if ".ipynb_checkpoints" not in str(pdf):
            shutil.copy(pdf, new_folder / (prefix + "_" + "report.pdf"))