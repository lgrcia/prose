\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{prose-report}[2021/01/01 prose-report]

\RequirePackage{calc}
\RequirePackage{changepage}
\LoadClass[10pt]{article}

% \DeclareOption{onecolumn}{\OptionNotUsed}
% \DeclareOption{green}{\renewcommand{\headlinecolor}{\color{green}}}
% \DeclareOption{red}{\renewcommand{\headlinecolor}{\color{slcolor}}}
% \DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
% \ProcessOptions\relax
% \LoadClass[twocolumn]{article}

\setlength{\footskip}{1pt}
\setlength{\parindent}{-0.1cm}


% defining new lengths
% ------------------------
\newcommand{\deflen}[2]{%      
    \expandafter\newlength\csname #1\endcsname
    \expandafter\setlength\csname #1\endcsname{#2}%
}

\deflen{marginsep}{1cm}
\deflen{marginwidth}{5cm}
\deflen{margintotal}{\marginwidth + \marginsep + \marginsep}

% Add easy doc kind setter
% ------------------------
\newcommand\dochead[1]{\renewcommand\@dochead{#1}}
\newcommand\@dochead{}


% box title
\newcommand\boxtitle[1]{
    {\color{gray!50}{
        {\fontfamily{lmss}\selectfont\footnotesize #1} 
        \vspace{-0.4cm}\newline
        \rule{\linewidth}{0.5pt}
    }\vspace{-0.6cm}\newline}
}

% Change title (left align)
% ------------------------
\renewcommand{\maketitle}{\bgroup\setlength{\parindent}{0pt}
\begin{flushleft}
    % \MakeUppercase
    {\textcolor{gray!50}{\small\textit{{\@dochead}}}\newline\vspace{-0.35cm}\newline}
    \kern-1em{\LARGE\@title}
    
  \vspace{-0.45cm}
  {\hspace{0.2em}\footnotesize\textit{\@author}}
  \vspace{-0.5cm}
\end{flushleft}\egroup
}

% env for figure with caption in margin
% ------------------------
\newenvironment{figurew}
    {\begin{figure}
        \begin{adjustwidth}{0cm}{-\marginwidth}}
    {   \end{adjustwidth}
    \end{figure}}

% Style caption
% ------------------------
\RequirePackage{caption}
\DeclareCaptionLabelFormat{bf-parens}{\textbf{#1 #2}}
\DeclareCaptionLabelSeparator{captionsep}{\vspace{4pt}\newline}
\captionsetup{labelformat=bf-parens,labelsep=captionsep, font=footnotesize,labelfont=footnotesize}

\newcommand{\acaption}[2]{
    \caption{
        {\fontfamily{qpl}\selectfont\scriptsize{\uppercase{#1}}}
        \vspace{-5pt}\newline\rule{\textwidth}{0.4pt}\newline
        #2
    }
}

% Margin notes
% ------------------------
\newcommand{\amarginnote}[1]{$^\cdot$\marginpar{$\cdot$ \footnotesize{#1}}}

% xxxxxxxxxxxxxxxxxxxxxxxxx

\captionsetup{justification=raggedright,singlelinecheck=false}
\linespread{1.3}
\setlength{\floatsep}{4cm}
\setlength{\parskip}{1em}
\setlength{\textwidth}{17.5cm}
\setlength{\textheight}{22cm}
\setcounter{secnumdepth}{0}

% GEOMETRY AND DOCUMENT INFO
% ------------------------
\RequirePackage[top=0.6cm, bottom=1cm, outer=1cm, inner=1cm, heightrounded, landscape]{geometry}
