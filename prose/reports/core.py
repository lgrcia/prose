import os
from os import path
import shutil
import jinja2
from .. import viz


template_folder = path.abspath(path.join(path.dirname(__file__), "..", "..", "latex"))


class LatexTemplate:
    def __init__(self, template_name, style="paper"):
        self.template_name = template_name
        self._style = style
        self.template = None
        self.dpi=150
        self.load_template()

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

    def make_report_folder(self, destination):
        if not path.exists(destination):
            os.mkdir(destination)

        self.destination = destination
        self.report_name = path.split(path.abspath(self.destination))[-1]
        self.measurement_destination = path.join(self.destination, f"{self.report_name}.txt")
        self.figure_destination = path.join(destination, "figures")
        if not path.exists(self.figure_destination):
            os.mkdir(self.figure_destination)


class Report(LatexTemplate):

    def __init__(self, reports, template_name="report.tex"):
        LatexTemplate.__init__(self, template_name)
        self.reports = reports
        self.paths = None

    def compile(self):
        cwd = os.getcwd()
        os.chdir(self.destination)
        os.system(f"pdflatex {self.report_name}")
        os.chdir(cwd)

    def make(self, destination):
        self.make_report_folder(destination)
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
