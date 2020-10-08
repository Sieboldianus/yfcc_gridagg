# Privacy-aware Flickr YFCC grid aggregation using HyperLogLog

This is the [git repository][static-gh-url] containing the supplementary materials for the paper

    Dunkel, A., Löchner, M., & Burghardt, D. (2020). Privacy-aware visualization
    of volunteered geographic information (VGI) to analyze spatial activity: A benchmark
    implementation. ISPRS International Journal of Geo-Information.

The notebooks are stored as markdown files with [jupytext][1] for better git compatibility.

These notebooks can be run with [jupyterlab-docker][2].

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
mkdir ./notebooks && mkdir ./py && jupytext --set-formats notebooks///ipynb,md///md,py///_/.py
```

Jupytext should automatically pick up the jupytext.toml configuration. 
If not, to sync notebooks (`./notebooks/`) automatically with markdown files (`./md/`) 
and python files (`./py/`), use:
```bash
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync notebooks/01_preparations.ipynb
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync notebooks/02_yfcc_gridagg_raw.ipynb
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync notebooks/03_yfcc_gridagg_hll.ipynb
jupytext --set-formats notebooks///ipynb,md///md,py///_/.py --sync notebooks/04_interpretation.ipynb
```

## Releases

* **v.0.1.0**: First release submitted alongside the paper.

[1]: https://github.com/mwouts/jupytext
[2]: https://gitlab.vgiscience.de/lbsn/tools/jupyterlab
[static-gh-url]: https://github.com/Sieboldianus/yfcc_gridagg