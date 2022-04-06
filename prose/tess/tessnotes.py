from ..reports.core import LatexTemplate


class TESSNotes(LatexTemplate):

    def __init__(self, style="paper", template_name="tess-notes.tex"):

        LatexTemplate.__init__(self, template_name, style=style)

    def make(self, destination):
        self.make_report_folder(destination)
        open(self.tex_destination, "w").write(self.template.render())