[![version](http://ad.vgiscience.org/yfcc_gridagg/version.svg)][static-gh-url] [![pipeline](http://ad.vgiscience.org/yfcc_gridagg/pipeline.svg)][static-gl-url] [![doi](http://ad.vgiscience.org/yfcc_gridagg/doi.svg)][DOI]

# Privacy-aware Flickr YFCC grid aggregation using HyperLogLog

![Cover Figures](https://raw.githubusercontent.com/Sieboldianus/yfcc_gridagg/master/resources/cover_figures.png)

This is the [git repository][static-gh-url] containing the supplementary materials for the paper

> **Dunkel, A., Löchner, M., & Burghardt, D. (2020).** _Privacy-aware visualization
  of volunteered geographic information (VGI) to analyze spatial activity: A benchmark
  implementation._ ISPRS International Journal of Geo-Information. [DOI][DOI-paper] / [PDF][PDF-paper]

The notebooks are stored as markdown files with [jupytext][1] for better git compatibility.

These notebooks can be run with [jupyterlab-docker][2].


## Results

In addition to the release file, latest HTML converts of notebooks are available here:

- [01_preparations.html](http://ad.vgiscience.org/yfcc_gridagg/01_preparations.html)
- [02_yfcc_gridagg_raw.html](http://ad.vgiscience.org/yfcc_gridagg/02_yfcc_gridagg_raw.html)
- [03_yfcc_gridagg_hll.html](http://ad.vgiscience.org/yfcc_gridagg/03_yfcc_gridagg_hll.html)
- [04_interpretation.html](http://ad.vgiscience.org/yfcc_gridagg/04_interpretation.html)

There are two additional HTMLs produced in notebooks:

- [yfcc_compare_raw_hll.html](http://ad.vgiscience.org/yfcc_gridagg/yfcc_compare_raw_hll.html) includes all figures and allows to compare raw/HLL through HTML tabs.
- [yfcc_usercount_est.html](http://ad.vgiscience.org/yfcc_gridagg/yfcc_usercount_est.html) interactive map of worldwide user count, for comparison of raw/HLL per bin.

## Convert to ipynb files

First, either download release files or convert the markdown files to working jupyter notebooks.

To convert jupytext markdown files:

If you're using the docker image, 
open a terminal inside jupyter and follow these commands:
```bash
bash
conda activate jupyter_env && cd /home/jovyan/work/
```

Afterwards, re-create the `.ipynb` notebook(s) with:
```bash
mkdir notebooks
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync md/01_preparations.md
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync md/02_yfcc_gridagg_raw.md
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync md/03_yfcc_gridagg_hll.md
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync md/04_interpretation.md
```

## Releases

* **v.0.1.1**: Minor refactor as part of the submission process
* **v.0.1.0**: First release submitted alongside the paper.

[1]: https://github.com/mwouts/jupytext
[2]: https://gitlab.vgiscience.de/lbsn/tools/jupyterlab
[static-gh-url]: https://github.com/Sieboldianus/yfcc_gridagg
[static-gl-url]: https://gitlab.vgiscience.de/ad/yfcc_gridagg
[DOI]: http://dx.doi.org/10.25532/OPARA-90
[DOI-paper]: https://doi.org/10.3390/ijgi9100607
[PDF-paper]: https://www.mdpi.com/2220-9964/9/10/607/pdf