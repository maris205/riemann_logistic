#!/usr/bin/env python3
# Transform revised SR body (body_src.tex) into MDPI numbered structure,
# and splice into main.tex at the %%% BODY_INSERT %%% marker.
import re, sys

SRC = "body_src.tex"
MAIN = "main.tex"

with open(SRC, encoding="utf-8") as f:
    txt = f.read()

# --- 1. isolate the body: from \section*{Introduction} up to (excluding) Code availability
m_start = txt.index(r"\section*{Introduction}")
m_end = txt.index(r"\section*{Code availability}")
body = txt[m_start:m_end]
tail = txt[m_end:]  # backmatter + supplementary, handled separately

# --- 2. de-star section headings (enables MDPI auto-numbering)
body = re.sub(r"\\section\*\{", r"\\section{", body)
body = re.sub(r"\\subsection\*\{", r"\\subsection{", body)
body = re.sub(r"\\subsubsection\*\{", r"\\subsubsection{", body)

# --- 2b. Fix the malformed "Effective lambda_n" equation (Reviewer 1, point 2).
#     Replace the broken double-\underset (with the undefined word "Effective")
#     by a clean \underbrace that DEFINES the effective control parameter.
BROKEN_EQ = (r"\[\text{x}_{\text{n}\text{+1}}\text{=1\ensuremath{-}}"
             r"\underset{\text{Effective}\text{\ensuremath{\lambda}}_{\text{n}}}"
             r"{\underset{\text{\ensuremath{\underbrace{\phantom{x}}}}}"
             r"{\left( \text{\ensuremath{\mu}}_{\text{c}}\text{+}"
             r"\frac{\text{k}_{\text{1}}}{\text{ln}^{\text{2}}\text{n}}\text{+}"
             r"\frac{\text{k}_{\text{2}}}{\text{ln}^{\text{3}}\text{n}} \right)}}"
             r"\text{x}_{\text{n}}^{\text{2}}\]")
CLEAN_EQ = (r"\begin{equation}\label{eq:eff}" "\n"
            r"x_{n+1} \;=\; 1 \;-\; "
            r"\underbrace{\left( \mu_c + \frac{k_1}{\ln^2 n} + \frac{k_2}{\ln^3 n} \right)}"
            r"_{\displaystyle \lambda_n^{\mathrm{eff}}}\, x_n^{2},"
            "\n" r"\end{equation}")
if BROKEN_EQ in body:
    body = body.replace(BROKEN_EQ, CLEAN_EQ)
    print("fixed: Effective equation")
    # insert an explicit definition sentence right after the fixed equation
    define_sentence = (
        r" Here $\lambda_n^{\mathrm{eff}}$ denotes the \emph{effective} "
        r"(iteration-dependent) control parameter actually applied at step $n$: "
        r"the constant baseline $\mu_c$ plus the two running corrections. "
        r"Equation~\eqref{eq:eff} thus makes explicit that "
        r"\eqref{eq:split} is realized with "
        r"$\delta_n = k_1/\ln^2 n + k_2/\ln^3 n$.")
    body = body.replace(
        CLEAN_EQ + "\n\n" + "This explicit construction",
        CLEAN_EQ + "\n" + define_sentence + "\n\n" + "This explicit construction",
        1)
else:
    print("WARN: broken Effective eq not found verbatim")

# also label the first governing map equation and the lambda_n split
body = body.replace(
    r"\[\text{x}_{\text{n}\text{+1}}\text{=1\ensuremath{-}}\text{\ensuremath{\lambda}}_{\text{n}}\text{x}_{\text{n}}^{\text{2}}\]",
    r"\begin{equation}\label{eq:map}" "\n" r"x_{n+1} = 1 - \lambda_n\, x_n^{2},"
    "\n" r"\end{equation}")
body = body.replace(
    r"\[\text{\ensuremath{\lambda}}_{\text{n}}\text{=}\text{\ensuremath{\mu}}_{\text{c}}\text{+}\text{\ensuremath{\delta}}_{\text{n}}\]",
    r"\begin{equation}\label{eq:split}" "\n" r"\lambda_n = \mu_c + \delta_n,"
    "\n" r"\end{equation}")

# clarify the "Effective Critical Baseline" wording and define lambda^eff in prose
body = body.replace(
    r"is identified as the" "\n" r"Effective Critical Baseline.",
    r"is a fixed baseline value of the control parameter (the ``critical baseline'').")
body = body.replace(
    "is identified as the Effective Critical Baseline.",
    "is a fixed baseline value of the control parameter (the ``critical baseline'').")

# --- 2c. strip a stray \bibliography{...} call that sits just before
#         "Code availability" in the original SR file (it got included in
#         the body slice); MDPI's own \bibliography call goes at the end.
body = re.sub(r"\\bibliography\{[^}]*\}\s*", "", body)

# --- 3. convert display math \[ ... \] to numbered equation environments.
#     Handle multiline. Non-greedy across the \[ \] pair.
def eq_repl(match):
    inner = match.group(1).strip()
    return "\\begin{equation}\n" + inner + "\n\\end{equation}"
body = re.sub(r"\\\[(.*?)\\\]", eq_repl, body, flags=re.DOTALL)

# --- 4. fix hardcoded cross-refs that no longer have those numbers
body = body.replace("(see Section 3.1)",
                    "(see the model definition below)")
body = body.replace("detailed in the Parameter Optimization Methods (Section 5.7)",
                    "detailed in the Parameter Optimization Methods (Appendix)")
body = body.replace("Section 3.1", "Section~\\ref{sec:model}")

# --- 5. label the two core model equations for referencing.
#     The non-autonomous map x_{n+1}=1-lambda_n x_n^2 (first equation env).
#     We tag the mathematical-model section for \ref.
body = body.replace(r"\section{The mathematical model}",
                    r"\section{The Mathematical Model}\label{sec:model}")

# --- 6. Table float: SR used longtable; MDPI prefers table. Leave as-is (works).

# --- 7. assemble backmatter separately (handled in main.tex), so drop 'tail'
#        supplementary section -> keep as an Appendix in MDPI
# Extract supplementary block for appendix
supp_idx = tail.index(r"\section*{Supplementary Material}")
supp = tail[supp_idx:]
supp = supp.replace(r"\section*{Supplementary Material}",
                    "\\appendix\n\\section{Supplementary Numerical Notes}")
supp = re.sub(r"\\subsection\*\{", r"\\subsection{", supp)
supp = re.sub(r"\\subsubsection\*\{", r"\\subsubsection{", supp)
supp = re.sub(r"\\\[(.*?)\\\]", eq_repl, supp, flags=re.DOTALL)

# strip body_src.tex's own trailing \end{document} (it belongs to the
# source file, not to the spliced fragment) before combining.
supp = re.sub(r"\\end\{document\}\s*$", "", supp.rstrip())

# combine body + appendix
full_body = body.rstrip() + "\n\n" + supp.strip() + "\n"

# --- 8. splice into main.tex
with open(MAIN, encoding="utf-8") as f:
    main = f.read()
main = main.replace("%%% BODY_INSERT %%%", full_body)

# insert MDPI backmatter commands before \end{document}
backmatter = r"""
\vspace{6pt}

\authorcontributions{L.W. is the sole author. L.W. conceived and designed the study, developed the theoretical framework and the non-autonomous dynamical model, wrote all simulation code, performed the numerical computations, analyzed and interpreted the results, prepared all figures, and wrote and reviewed the manuscript. The author has read and agreed to the published version of the manuscript.}

\funding{This research received no external funding.}

\providecommand{\dataavailability}[1]{\vspace{6pt}\noindent{\fontsize{9}{9}\selectfont\textbf{Data Availability Statement:} {#1}\par}}
\dataavailability{All code required to reproduce the results is openly available at \url{https://github.com/maris205/riemann_logistic}. The reference Riemann zeros used for benchmarking are obtained from standard high-precision tabulations (e.g., via \texttt{mpmath.zetazero}); no restricted or proprietary data were used.}

\acknowledgments{The author acknowledges the assistance of a large language model (Google Gemini~3 Pro) in code implementation for the numerical simulations and in auxiliary analysis and language editing during manuscript preparation. All scientific claims, results, and their interpretation are the responsibility of the author.}

\conflictsofinterest{The author declares no conflict of interest.}
"""
assert main.count(r"\end{document}") == 1, "expected exactly one \\end{document} before splicing backmatter"
main = main.replace(r"\end{document}", backmatter + "\n\\reftitle{References}\n\\externalbibliography{yes}\n\\bibliography{references}\n\n\\end{document}")

with open(MAIN, "w", encoding="utf-8") as f:
    f.write(main)

# report
neq = full_body.count(r"\begin{equation}")
nsec = len(re.findall(r"\\section\{", main))
print(f"equations numbered: {neq}")
print(f"\\section{{}} count: {nsec}")
print("done")
