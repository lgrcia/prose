_all_citations = {
"prose": """
@ARTICLE{prose,
       author = {{Garcia}, Lionel J. and {Timmermans}, Mathilde and {Pozuelos}, Francisco J. and {Ducrot}, Elsa and {Gillon}, Micha{\"e}l and {Delrez}, Laetitia and {Wells}, Robert D. and {Jehin}, Emmanu{\"e}l},
        title = "{PROSE: a PYTHON framework for modular astronomical images processing}",
      journal = {\mnras},
     keywords = {instrumentation: detectors, methods: data analysis, planetary systems, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
         year = 2022,
        month = feb,
       volume = {509},
       number = {4},
        pages = {4817-4828},
          doi = {10.1093/mnras/stab3113},
archivePrefix = {arXiv},
       eprint = {2111.02814},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.4817G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
"scipy": """
@article{scipy,
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
@misc{photutils,
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
"scikit-image": """
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
@misc{tensorflow,
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
@book{numpy,
 title={A guide to NumPy},
 author={Oliphant, Travis E},
 volume={1},
 year={2006},
 publisher={Trelgol Publishing USA}
}
""",
"source extractor": """
@article{sep0,
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
@article{sep,
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
""",
"astropy": """
@ARTICLE{astropy,
       author = {{Astropy Collaboration} and {Price-Whelan}, A.~M. and
         {Sip{\H{o}}cz}, B.~M. and {G{\"u}nther}, H.~M. and {Lim}, P.~L. and
         {Crawford}, S.~M. and {Conseil}, S. and {Shupe}, D.~L. and
         {Craig}, M.~W. and {Dencheva}, N. and {Ginsburg}, A. and {Vand
        erPlas}, J.~T. and {Bradley}, L.~D. and {P{\'e}rez-Su{\'a}rez}, D. and
         {de Val-Borro}, M. and {Aldcroft}, T.~L. and {Cruz}, K.~L. and
         {Robitaille}, T.~P. and {Tollerud}, E.~J. and {Ardelean}, C. and
         {Babej}, T. and {Bach}, Y.~P. and {Bachetti}, M. and {Bakanov}, A.~V. and
         {Bamford}, S.~P. and {Barentsen}, G. and {Barmby}, P. and
         {Baumbach}, A. and {Berry}, K.~L. and {Biscani}, F. and {Boquien}, M. and
         {Bostroem}, K.~A. and {Bouma}, L.~G. and {Brammer}, G.~B. and
         {Bray}, E.~M. and {Breytenbach}, H. and {Buddelmeijer}, H. and
         {Burke}, D.~J. and {Calderone}, G. and {Cano Rodr{\'\i}guez}, J.~L. and
         {Cara}, M. and {Cardoso}, J.~V.~M. and {Cheedella}, S. and {Copin}, Y. and
         {Corrales}, L. and {Crichton}, D. and {D'Avella}, D. and {Deil}, C. and
         {Depagne}, {\'E}. and {Dietrich}, J.~P. and {Donath}, A. and
         {Droettboom}, M. and {Earl}, N. and {Erben}, T. and {Fabbro}, S. and
         {Ferreira}, L.~A. and {Finethy}, T. and {Fox}, R.~T. and
         {Garrison}, L.~H. and {Gibbons}, S.~L.~J. and {Goldstein}, D.~A. and
         {Gommers}, R. and {Greco}, J.~P. and {Greenfield}, P. and
         {Groener}, A.~M. and {Grollier}, F. and {Hagen}, A. and {Hirst}, P. and
         {Homeier}, D. and {Horton}, A.~J. and {Hosseinzadeh}, G. and {Hu}, L. and
         {Hunkeler}, J.~S. and {Ivezi{\'c}}, {\v{Z}}. and {Jain}, A. and
         {Jenness}, T. and {Kanarek}, G. and {Kendrew}, S. and {Kern}, N.~S. and
         {Kerzendorf}, W.~E. and {Khvalko}, A. and {King}, J. and {Kirkby}, D. and
         {Kulkarni}, A.~M. and {Kumar}, A. and {Lee}, A. and {Lenz}, D. and
         {Littlefair}, S.~P. and {Ma}, Z. and {Macleod}, D.~M. and
         {Mastropietro}, M. and {McCully}, C. and {Montagnac}, S. and
         {Morris}, B.~M. and {Mueller}, M. and {Mumford}, S.~J. and {Muna}, D. and
         {Murphy}, N.~A. and {Nelson}, S. and {Nguyen}, G.~H. and
         {Ninan}, J.~P. and {N{\"o}the}, M. and {Ogaz}, S. and {Oh}, S. and
         {Parejko}, J.~K. and {Parley}, N. and {Pascual}, S. and {Patil}, R. and
         {Patil}, A.~A. and {Plunkett}, A.~L. and {Prochaska}, J.~X. and
         {Rastogi}, T. and {Reddy Janga}, V. and {Sabater}, J. and
         {Sakurikar}, P. and {Seifert}, M. and {Sherbert}, L.~E. and
         {Sherwood-Taylor}, H. and {Shih}, A.~Y. and {Sick}, J. and
         {Silbiger}, M.~T. and {Singanamalla}, S. and {Singer}, L.~P. and
         {Sladen}, P.~H. and {Sooley}, K.~A. and {Sornarajah}, S. and
         {Streicher}, O. and {Teuben}, P. and {Thomas}, S.~W. and
         {Tremblay}, G.~R. and {Turner}, J.~E.~H. and {Terr{\'o}n}, V. and
         {van Kerkwijk}, M.~H. and {de la Vega}, A. and {Watkins}, L.~L. and
         {Weaver}, B.~A. and {Whitmore}, J.~B. and {Woillez}, J. and
         {Zabalza}, V. and {Astropy Contributors}},
        title = "{The Astropy Project: Building an Open-science Project and Status of the v2.0 Core Package}",
      journal = {\aj},
     keywords = {methods: data analysis, methods: miscellaneous, methods: statistical, reference systems, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2018,
        month = sep,
       volume = {156},
       number = {3},
          eid = {123},
        pages = {123},
          doi = {10.3847/1538-3881/aabc4f},
archivePrefix = {arXiv},
       eprint = {1801.02634},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018AJ....156..123A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
}
