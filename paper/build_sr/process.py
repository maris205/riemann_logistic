import re, os

base = '/root/autodl-tmp/henon_mapping/riemann_logistic/mvp4/paper/build_sr'
raw = open(os.path.join(base, 'body_raw.tex')).read()

# ---- citation map (order matches the numbered reference list) ----
keys = ['riemann1859','montgomery1973','odlyzko1987','berry1986','berry1999',
        'connes1999','weyl1912','may1976','feigenbaum1978','arpack1998','gutzwiller1990']
def cite_sub(m):
    nums = re.findall(r'\d+', m.group(0))
    ks = [keys[int(n)-1] for n in nums if 1 <= int(n) <= len(keys)]
    return '\\cite{' + ','.join(ks) + '}' if ks else m.group(0)
raw = re.sub(r'\{\[\}[0-9][0-9, \-]*\{\]\}', cite_sub, raw)

# ---- brace-matching helper ----
def find_group(s, i):
    """s[i] must be '{'; return (content, index_after_closing_brace)."""
    assert s[i] == '{'
    depth = 0; j = i
    while j < len(s):
        if s[j] == '{': depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return s[i+1:j], j+1
        j += 1
    raise ValueError("unbalanced")

def clean_title(t):
    # drop inline math, \textbf, leftover pandoc escapes; strip numbering
    t = re.sub(r'\\\(.*?\\\)', '', t, flags=re.DOTALL)
    t = re.sub(r'\\textbf\{(.*?)\}', r'\1', t, flags=re.DOTALL)
    t = t.replace('\\textbackslash', '')
    # drop leftover escaped pandoc math remnants like  text\{U\}_\{...\}
    t = re.sub(r'text\\\{.*', '', t)
    t = t.replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'^(S\.)?\d+(\.\d+)*\s+', '', t)
    return t

# ---- parse headings sequentially via brace matching ----
out = []
i = 0
heading_re = re.compile(r'\\(section|subsection|subsubsection|paragraph|subparagraph)\{')
while i < len(raw):
    m = heading_re.match(raw, i)
    if not m:
        out.append(raw[i]); i += 1; continue
    level = m.group(1)
    brace_i = m.end() - 1            # position of '{'
    arg, after = find_group(raw, brace_i)   # whole {...} after \section
    # arg may be \texorpdfstring{A}{B}  -> use B (plain). else use arg directly.
    tp = re.match(r'\\texorpdfstring\{', arg)
    if tp:
        a1, k = find_group(arg, tp.end()-1)
        # second group
        a2, _ = find_group(arg, k)        # k points at '{' of 2nd arg
        title = clean_title(a2 if a2.strip() else a1)
    else:
        title = clean_title(arg)
    # skip a trailing \label{...} that may follow
    rest = raw[after:]
    lab = re.match(r'\s*\\label\{[^}]*\}', rest)
    if lab:
        after += lab.end()
    # map level -> SR (section/subsection are top sections*, deeper -> subsection*/subsubsection*)
    if level in ('section', 'subsection'):
        out.append('\\section*{%s}' % title)
    elif level == 'subsubsection':
        out.append('\\subsection*{%s}' % title)
    else:
        out.append('\\subsubsection*{%s}' % title)
    i = after

text = ''.join(out)

# ---- strip pandoc hypertarget wrappers ----
text = re.sub(r'\\hypertarget\{[^}]*\}\{%\s*', '', text)
# leftover lone '}' lines from closed hypertargets
text = re.sub(r'\n\}\s*\n', '\n\n', text)
# headings carry a trailing stray '}' that used to close the hypertarget group
text = re.sub(r'(\\(?:sub)*section\*\{[^{}]*\})\}', r'\1', text)

# ---- slice from Introduction to References; keep back-matter after refs ----
intro = text.index(r'\section*{Introduction}')
refs  = text.index(r'\section*{References}')
ack   = text.index(r'\section*{Acknowledgments}')
text = text[intro:refs] + '\n' + text[ack:]

# ---- image paths + sizing ----
text = text.replace('build_sr/media/', 'figures/')
text = re.sub(r'\\includegraphics\[[^\]]*\]\{figures/', r'\\includegraphics[width=\\linewidth]{figures/', text)

# ---- wrap "\includegraphics{...}\n\n\begin{quote}\textbf{Figure N. ...}...\end{quote}"
#      into a proper figure float with \caption ----
def figify(m):
    img = m.group(1)
    cap = m.group(2).strip()
    # turn leading "\textbf{Figure N. Title}" into caption text (keep bold title inline)
    cap = re.sub(r'\s+', ' ', cap)
    return ('\\begin{figure}[ht]\n\\centering\n'
            '\\includegraphics[width=\\linewidth]{figures/%s}\n'
            '\\caption{%s}\n\\end{figure}' % (img, cap))

text = re.sub(
    r'\\includegraphics\[width=\\linewidth\]\{figures/([^}]+)\}\s*\\begin\{quote\}(.*?)\\end\{quote\}',
    figify, text, flags=re.DOTALL)

# second pattern: image lives INSIDE the quote block, before the caption
def figify2(m):
    img = m.group(1); cap = m.group(2).strip()
    cap = re.sub(r'\s+', ' ', cap)
    return ('\\begin{figure}[ht]\n\\centering\n'
            '\\includegraphics[width=\\linewidth]{figures/%s}\n'
            '\\caption{%s}\n\\end{figure}' % (img, cap))
text = re.sub(
    r'\\begin\{quote\}\s*\\includegraphics\[width=\\linewidth\]\{figures/([^}]+)\}\s*(.*?)\\end\{quote\}',
    figify2, text, flags=re.DOTALL)

# tidy pandoc caption artifacts inside captions
text = text.replace('\\textgreater{}', '').replace('\\textgreater', '')

open(os.path.join(base, 'body_clean.tex'), 'w').write(text)
print("lines:", text.count(chr(10)))
print("cites:", len(re.findall(r'\\cite\{', text)))
print("section*:", len(re.findall(r'\\section\*', text)))
print("subsection*:", len(re.findall(r'\\subsection\*', text)))
print("includegraphics:", len(re.findall(r'\\includegraphics', text)))
print("=== headings ===")
for h in re.findall(r'\\(?:sub)?section\*\{[^}]*\}', text):
    print(" ", h)
