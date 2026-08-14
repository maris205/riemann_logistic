import re, os
base = '/root/autodl-tmp/henon_mapping/riemann_logistic/mvp4/paper/build_sr'

uni = {
 '±': r'\ensuremath{\pm}', '×': r'\ensuremath{\times}', 'ï': r'\"\i{}', 'ó': r"\'o",
 'Δ': r'\ensuremath{\Delta}', 'Φ': r'\ensuremath{\Phi}', 'Ω': r'\ensuremath{\Omega}',
 'β': r'\ensuremath{\beta}', 'γ': r'\ensuremath{\gamma}', 'δ': r'\ensuremath{\delta}',
 'ζ': r'\ensuremath{\zeta}', 'θ': r'\ensuremath{\theta}', 'λ': r'\ensuremath{\lambda}',
 'μ': r'\ensuremath{\mu}', 'π': r'\ensuremath{\pi}', 'ρ': r'\ensuremath{\rho}',
 'σ': r'\ensuremath{\sigma}', 'ϵ': r'\ensuremath{\epsilon}', '…': r'\ldots{}',
 '→': r'\ensuremath{\rightarrow}', '∈': r'\ensuremath{\in}', '∏': r'\ensuremath{\prod}',
 '−': r'\ensuremath{-}', '∝': r'\ensuremath{\propto}', '∞': r'\ensuremath{\infty}',
 '∼': r'\ensuremath{\sim}', '≈': r'\ensuremath{\approx}', '≡': r'\ensuremath{\equiv}',
 '≤': r'\ensuremath{\le}', '≥': r'\ensuremath{\ge}', '≫': r'\ensuremath{\gg}',
 '⋅': r'\ensuremath{\cdot}', '⏟': r'\ensuremath{\underbrace{\phantom{x}}}',
}
def demap(s):
    for k,v in uni.items(): s = s.replace(k,v)
    return s

abstract = demap(open(os.path.join(base,'abstract.txt'),encoding='utf-8').read().strip())
body = demap(open(os.path.join(base,'body_clean.tex'),encoding='utf-8').read())

# split body: main text up to "Code availability"; that and after = back matter
cut = body.index(r'\section*{Code availability}')
maintext = body[:cut].rstrip()
backmatter = body[cut:]

# In SR template, Methods is the last numbered section; back-matter uses \section*
# We'll place maintext (Intro/Model/Results/Discussion/Methods) then bibliography then backmatter.

preamble = r'''\documentclass[fleqn,10pt]{wlscirep}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{longtable,booktabs,array}
\usepackage{float}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

\title{Spectral Isomorphism between Renormalization Flow in Non-Autonomous Quadratic Maps and Riemann Zeros}

\author[1,*]{Liang Wang}
\affil[1]{School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan, 430070, P.R. China}
\affil[*]{wangliang.f@gmail.com}

\begin{abstract}
%ABSTRACT%
\end{abstract}
\begin{document}

\flushbottom
\maketitle
\thispagestyle{empty}

%MAINTEXT%

\bibliography{references}

%BACKMATTER%

\end{document}
'''

doc = (preamble
       .replace('%ABSTRACT%', abstract)
       .replace('%MAINTEXT%', maintext)
       .replace('%BACKMATTER%', backmatter))
open(os.path.join(base,'main.tex'),'w').write(doc)
print("main.tex written:", len(doc), "chars")
print("remaining non-ascii:", sorted({c for c in doc if ord(c)>127}))
