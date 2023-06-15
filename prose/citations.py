import astropy

citations = {
    "numpy": """
@Article{numpy,
 title = {Array programming with {NumPy}},
 author = {Charles R. Harris and K. Jarrod Millman and St{\'{e}}fan J.
        van der Walt and Ralf Gommers and Pauli Virtanen and David
        Cournapeau and Eric Wieser and Julian Taylor and Sebastian
        Berg and Nathaniel J. Smith and Robert Kern and Matti Picus
        and Stephan Hoyer and Marten H. van Kerkwijk and Matthew
        Brett and Allan Haldane and Jaime Fern{\'{a}}ndez del
        R{\'{i}}o and Mark Wiebe and Pearu Peterson and Pierre
        G{\'{e}}rard-Marchant and Kevin Sheppard and Tyler Reddy and
        Warren Weckesser and Hameer Abbasi and Christoph Gohlke and
        Travis E. Oliphant},
 year = {2020},
 month = sep,
 journal = {Nature},
 volume = {585},
 number = {7825},
 pages = {357--362},
 doi = {10.1038/s41586-020-2649-2},
 publisher = {Springer Science and Business Media {LLC}},
 url = {https://doi.org/10.1038/s41586-020-2649-2}
}
""",
    "astropy": """
    @ARTICLE{astropy,
       author = {{Astropy Collaboration} and {Price-Whelan}, Adrian M. and {Lim}, Pey
       Lian and {Earl}, Nicholas and {Starkman}, Nathaniel and {Bradley}, Larry and
       {Shupe}, David L. and {Patil}, Aarya A. and {Corrales}, Lia and {Brasseur}, C.~E.
       and {N{\"o}the}, Maximilian and {Donath}, Axel and {Tollerud}, Erik and {Morris},
       Brett M. and {Ginsburg}, Adam and {Vaher}, Eero and {Weaver}, Benjamin A. and
       {Tocknell}, James and {Jamieson}, William and {van Kerkwijk}, Marten H. and
       {Robitaille}, Thomas P. and {Merry}, Bruce and {Bachetti}, Matteo and
       {G{\"u}nther}, H. Moritz and {Aldcroft}, Thomas L. and {Alvarado-Montes}, Jaime
       A. and {Archibald}, Anne M. and {B{\'o}di}, Attila and {Bapat}, Shreyas and
       {Barentsen}, Geert and {Baz{\'a}n}, Juanjo and {Biswas}, Manish and {Boquien},
       M{\'e}d{\'e}ric and {Burke}, D.~J. and {Cara}, Daria and {Cara}, Mihai and
       {Conroy}, Kyle E. and {Conseil}, Simon and {Craig}, Matthew W. and {Cross},
       Robert M. and {Cruz}, Kelle L. and {D'Eugenio}, Francesco and {Dencheva}, Nadia
       and {Devillepoix}, Hadrien A.~R. and {Dietrich}, J{\"o}rg P. and {Eigenbrot},
       Arthur Davis and {Erben}, Thomas and {Ferreira}, Leonardo and {Foreman-Mackey},
       Daniel and {Fox}, Ryan and {Freij}, Nabil and {Garg}, Suyog and {Geda}, Robel and
       {Glattly}, Lauren and {Gondhalekar}, Yash and {Gordon}, Karl D. and {Grant},
       David and {Greenfield}, Perry and {Groener}, Austen M. and {Guest}, Steve and
       {Gurovich}, Sebastian and {Handberg}, Rasmus and {Hart}, Akeem and
       {Hatfield-Dodds}, Zac and {Homeier}, Derek and {Hosseinzadeh}, Griffin and
       {Jenness}, Tim and {Jones}, Craig K. and {Joseph}, Prajwel and {Kalmbach}, J.
       Bryce and {Karamehmetoglu}, Emir and {Ka{\l}uszy{\'n}ski}, Miko{\l}aj and
       {Kelley}, Michael S.~P. and {Kern}, Nicholas and {Kerzendorf}, Wolfgang E. and
       {Koch}, Eric W. and {Kulumani}, Shankar and {Lee}, Antony and {Ly}, Chun and
       {Ma}, Zhiyuan and {MacBride}, Conor and {Maljaars}, Jakob M. and {Muna}, Demitri
       and {Murphy}, N.~A. and {Norman}, Henrik and {O'Steen}, Richard and {Oman}, Kyle
       A. and {Pacifici}, Camilla and {Pascual}, Sergio and {Pascual-Granado}, J. and
       {Patil}, Rohit R. and {Perren}, Gabriel I. and {Pickering}, Timothy E. and
       {Rastogi}, Tanuj and {Roulston}, Benjamin R. and {Ryan}, Daniel F. and {Rykoff},
       Eli S. and {Sabater}, Jose and {Sakurikar}, Parikshit and {Salgado}, Jes{\'u}s
       and {Sanghi}, Aniket and {Saunders}, Nicholas and {Savchenko}, Volodymyr and
       {Schwardt}, Ludwig and {Seifert-Eckert}, Michael and {Shih}, Albert Y. and
       {Jain}, Anany Shrey and {Shukla}, Gyanendra and {Sick}, Jonathan and {Simpson},
       Chris and {Singanamalla}, Sudheesh and {Singer}, Leo P. and {Singhal}, Jaladh and
       {Sinha}, Manodeep and {Sip{\H{o}}cz}, Brigitta M. and {Spitler}, Lee R. and
       {Stansby}, David and {Streicher}, Ole and {{\v{S}}umak}, Jani and {Swinbank},
       John D. and {Taranu}, Dan S. and {Tewary}, Nikita and {Tremblay}, Grant R. and
       {Val-Borro}, Miguel de and {Van Kooten}, Samuel J. and {Vasovi{\'c}}, Zlatan and
       {Verma}, Shresth and {de Miranda Cardoso}, Jos{\'e} Vin{\'\i}cius and {Williams},
       Peter K.~G. and {Wilson}, Tom J. and {Winkel}, Benjamin and {Wood-Vasey}, W.~M.
       and {Xue}, Rui and {Yoachim}, Peter and {Zhang}, Chen and {Zonca}, Andrea and
       {Astropy Project Contributors}}, title = "{The Astropy Project: Sustaining and
       Growing a Community-oriented Open-source Project and the Latest Major Release
       (v5.0) of the Core Package}",
      journal = {\apj},
     keywords = {Astronomy software, Open source software, Astronomy data analysis, 1855, 1866, 1858, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = aug,
       volume = {935},
       number = {2},
          eid = {167},
        pages = {167},
          doi = {10.3847/1538-4357/ac7c74},
archivePrefix = {arXiv},
       eprint = {2206.14220},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJ...935..167A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    "scipy": """
@ARTICLE{scipy,
  author  = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
            Haberland, Matt and Reddy, Tyler and Cournapeau, David and
            Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
            Bright, Jonathan and {van der Walt}, St{\'e}fan J. and
            Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
            Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
            Kern, Robert and Larson, Eric and Carey, C J and
            Polat, {\.I}lhan and Feng, Yu and Moore, Eric W. and
            {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
            Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
            Harris, Charles R. and Archibald, Anne M. and
            Ribeiro, Ant{\^o}nio H. and Pedregosa, Fabian and
            {van Mulbregt}, Paul and {SciPy 1.0 Contributors}},
  title   = {{{SciPy} 1.0: Fundamental Algorithms for Scientific
            Computing in Python}},
  journal = {Nature Methods},
  year    = {2020},
  volume  = {17},
  pages   = {261--272},
  adsurl  = {https://rdcu.be/b08Wh},
  doi     = {10.1038/s41592-019-0686-2},
}
""",
    "scikit-image": """
@article{scikit-image,
 title = {scikit-image: image processing in {P}ython},
 author = {van der Walt, {S}t\'efan and {S}ch\"onberger, {J}ohannes {L}. and
           {Nunez-Iglesias}, {J}uan and {B}oulogne, {F}ran\c{c}ois and {W}arner,
           {J}oshua {D}. and {Y}ager, {N}eil and {G}ouillart, {E}mmanuelle and
           {Y}u, {T}ony and the scikit-image contributors},
 year = {2014},
 month = {6},
 keywords = {Image processing, Reproducible research, Education,
             Visualization, Open source, Python, Scientific programming},
 volume = {2},
 pages = {e453},
 journal = {PeerJ},
 issn = {2167-8359},
 url = {https://doi.org/10.7717/peerj.453},
 doi = {10.7717/peerj.453}
}
""",
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
    "photutils": """
@software{photutils,
author       = {Larry Bradley and
                Brigitta Sipőcz and
                Thomas Robitaille and
                Erik Tollerud and
                Zé Vinícius and
                Christoph Deil and
                Kyle Barbary and
                Tom J Wilson and
                Ivo Busko and
                Axel Donath and
                Hans Moritz Günther and
                Mihai Cara and
                P. L. Lim and
                Sebastian Meßlinger and
                Simon Conseil and
                Azalee Bostroem and
                Michael Droettboom and
                E. M. Bray and
                Lars Andersen Bratholm and
                Geert Barentsen and
                Matt Craig and
                Shivangee Rathi and
                Sergio Pascual and
                Gabriel Perren and
                Iskren Y. Georgiev and
                Miguel de Val-Borro and
                Wolfgang Kerzendorf and
                Yoonsoo P. Bach and
                Bruno Quint and
                Harrison Souchereau},
title        = {astropy/photutils: 1.5.0},
month        = jul,
year         = 2022,
publisher    = {Zenodo},
version      = {1.5.0},
doi          = {10.5281/zenodo.6825092},
url          = {https://doi.org/10.5281/zenodo.6825092}
}
""",
    "astroquery": """
@ARTICLE{astroquery,
   author = {{Ginsburg}, A. and {Sip{\H o}cz}, B.~M. and {Brasseur}, C.~E. and
	{Cowperthwaite}, P.~S. and {Craig}, M.~W. and {Deil}, C. and
	{Guillochon}, J. and {Guzman}, G. and {Liedtke}, S. and {Lian Lim}, P. and
	{Lockhart}, K.~E. and {Mommert}, M. and {Morris}, B.~M. and
	{Norman}, H. and {Parikh}, M. and {Persson}, M.~V. and {Robitaille}, T.~P. and
	{Segovia}, J.-C. and {Singer}, L.~P. and {Tollerud}, E.~J. and
	{de Val-Borro}, M. and {Valtchanov}, I. and {Woillez}, J. and
	{The Astroquery collaboration} and {a subset of the astropy collaboration}
	},
    title = "{astroquery: An Astronomical Web-querying Package in Python}",
  journal = {\aj},
archivePrefix = "arXiv",
   eprint = {1901.04520},
 primaryClass = "astro-ph.IM",
 keywords = {astronomical databases: miscellaneous, virtual observatory tools},
     year = 2019,
    month = mar,
   volume = 157,
      eid = {98},
    pages = {98},
      doi = {10.3847/1538-3881/aafc33},
   adsurl = {http://adsabs.harvard.edu/abs/2019AJ....157...98G},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
}
