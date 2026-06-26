# Paper Report

This directory contains the polished paper-style report for the fisheye motion project.

## Files

- `main.tex`: CVPR/ICCV-style two-column Chinese LaTeX report.
- `references.bib`: bibliography used by the report.
- `../figures/`: generated paper figures.

## Build

Install a LaTeX distribution with Chinese support, then run:

```bash
cd docs/paper
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

The source uses `ctexart`, so `xelatex` is recommended.

On the original Ubuntu workstation, XeLaTeX was installed without sudo under:

```bash
/home/ferdinand/texlive/2026
```

Use it in the current shell with:

```bash
export PATH=/home/ferdinand/texlive/2026/bin/x86_64-linux:$PATH
```

The resulting compiled PDF is `main.pdf`.

If the figures need to be regenerated from experiment outputs:

```bash
cd ../..
python scripts/make_academic_report_assets.py
```
