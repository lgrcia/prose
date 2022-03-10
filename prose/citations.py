from .. import Block, Sequence


def get_citations(*args):
    """
    Returns citation text and bib from a list of units or blocks (Inspired by :code:`exoplanet` from dfm)

    Parameters
    ----------
    args

    Returns
    -------

    """
    citations = {}
    for block_or_unit in args:
        assert isinstance(block_or_unit, (Block, Sequence)), "args should be units or blocks"
        block_or_unit_citations = block_or_unit.citations()
        if block_or_unit_citations is not None:
            for citation in block_or_unit_citations:
                citations[citation] = _all_citations[citation]

    txt = "This research made use of textsf{{prose}} citep{{prose}} and its dependencies citep{{{}}}.""".format(
        ", ".join(["prose:{}".format(citation) for citation in citations.keys()]))

    bib = "\n\n".join(citations.values())

    return txt, bib


_all_citations = {
"prose": """
TODO
""",
"scipy": """
@article{2020SciPy-NMeth,
       author = {{Virtanen}, Pauli and {Gommers}, Ralf and {Oliphant},
         Travis E. and {Haberland}, Matt and {Reddy}, Tyler and
         {Cournapeau}, David and {Burovski}, Evgeni and {Peterson}, Pearu
         and {Weckesser}, Warren and {Bright}, Jonathan and {van der Walt},
         St{\'e}fan J.  and {Brett}, Matthew and {Wilson}, Joshua and
         {Jarrod Millman}, K.  and {Mayorov}, Nikolay and {Nelson}, Andrew
         R.~J. and {Jones}, Eric and {Kern}, Robert and {Larson}, Eric and
         {Carey}, CJ and {Polat}, {\.I}lhan and {Feng}, Yu and {Moore},
         Eric W. and {Vand erPlas}, Jake and {Laxalde}, Denis and
         {Perktold}, Josef and {Cimrman}, Robert and {Henriksen}, Ian and
         {Quintero}, E.~A. and {Harris}, Charles R and {Archibald}, Anne M.
         and {Ribeiro}, Ant{\^o}nio H. and {Pedregosa}, Fabian and
         {van Mulbregt}, Paul and {Contributors}, SciPy 1. 0},
        title = "{SciPy 1.0: Fundamental Algorithms for Scientific
                  Computing in Python}",
      journal = {Nature Methods},
      year = "2020",
      volume={17},
      pages={261--272},
      adsurl = {https://rdcu.be/b08Wh},
      doi = {https://doi.org/10.1038/s41592-019-0686-2},
}
""",
"photutils": """
@misc{Bradley_2019_2533376,
   author = {Larry Bradley and Brigitta Sip{\H o}cz and Thomas Robitaille and
             Erik Tollerud and Z\`e Vin{\'{\i}}cius and Christoph Deil and
             Kyle Barbary and Hans Moritz G{\"u}nther and Mihai Cara and
             Ivo Busko and Simon Conseil and Michael Droettboom and
             Azalee Bostroem and E. M. Bray and Lars Andersen Bratholm and
             Tom Wilson and Matt Craig and Geert Barentsen and
             Sergio Pascual and Axel Donath and Johnny Greco and
             Gabriel Perren and P. L. Lim and Wolfgang Kerzendorf},
    title = {astropy/photutils: v0.6},
    month = jan,
     year = 2019,
      doi = {10.5281/zenodo.2533376},
      url = {https://doi.org/10.5281/zenodo.2533376}
}
""",
"skimage": """
@article{scikit-image,
	Author = {van der Walt, {S}t\'efan and {S}ch\"onberger, {J}ohannes {L}. and {Nunez-Iglesias}, {J}uan and {B}oulogne, {F}ran\c{c}ois and {W}arner, {J}oshua {D}. and {Y}ager, {N}eil and {G}ouillart, {E}mmanuelle and {Y}u, {T}ony and the scikit-image contributors},
	Doi = {10.7717/peerj.453},
	Issn = {2167-8359},
	Journal = {PeerJ},
	Keywords = {Image processing, Reproducible research, Education, Visualization, Open source, Python, Scientific programming},
	Month = {6},
	Pages = {e453},
	Title = {scikit-image: image processing in {P}ython},
	Url = {https://doi.org/10.7717/peerj.453},
	Volume = {2},
	Year = {2014},
	Bdsk-Url-1 = {https://doi.org/10.7717/peerj.453}}
""",
"tensorflow": """
@misc{tensorflow2015-whitepaper,
title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
url={https://www.tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart\'{\i}n~Abadi and
    Ashish~Agarwal and
    Paul~Barham and
    Eugene~Brevdo and
    Zhifeng~Chen and
    Craig~Citro and
    Greg~S.~Corrado and
    Andy~Davis and
    Jeffrey~Dean and
    Matthieu~Devin and
    Sanjay~Ghemawat and
    Ian~Goodfellow and
    Andrew~Harp and
    Geoffrey~Irving and
    Michael~Isard and
    Yangqing Jia and
    Rafal~Jozefowicz and
    Lukasz~Kaiser and
    Manjunath~Kudlur and
    Josh~Levenberg and
    Dandelion~Man\'{e} and
    Rajat~Monga and
    Sherry~Moore and
    Derek~Murray and
    Chris~Olah and
    Mike~Schuster and
    Jonathon~Shlens and
    Benoit~Steiner and
    Ilya~Sutskever and
    Kunal~Talwar and
    Paul~Tucker and
    Vincent~Vanhoucke and
    Vijay~Vasudevan and
    Fernanda~Vi\'{e}gas and
    Oriol~Vinyals and
    Pete~Warden and
    Martin~Wattenberg and
    Martin~Wicke and
    Yuan~Yu and
    Xiaoqiang~Zheng},
  year={2015},
}
""",
"numpy": """
@book{oliphant2006guide,
 title={A guide to NumPy},
 author={Oliphant, Travis E},
 volume={1},
 year={2006},
 publisher={Trelgol Publishing USA}
}
""",
"source extractor": """
@article{ refId0,
	author = {{Bertin, E.} and {Arnouts, S.}},
	title = {SExtractor: Software for source extraction},
	DOI= "10.1051/aas:1996164",
	url= "https://doi.org/10.1051/aas:1996164",
	journal = {Astron. Astrophys. Suppl. Ser.},
	year = 1996,
	volume = 117,
	number = 2,
	pages = "393-404",
}
""",
"sep": """
@article{Barbary2016,
  doi = {10.21105/joss.00058},
  url = {https://doi.org/10.21105/joss.00058},
  year = {2016},
  publisher = {The Open Journal},
  volume = {1},
  number = {6},
  pages = {58},
  author = {Kyle Barbary},
  title = {SEP: Source Extractor as a library},
  journal = {Journal of Open Source Software}
}

"""
}
