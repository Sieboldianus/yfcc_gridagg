# Privacy-aware Flickr YFCC grid aggregation using HyperLogLog

The notebooks are stored as markdown files with [jupytext][1] for better git compatibility.

These notebooks can be run with [jupyterlab-docker][2].

First, either download release files or convert the markdown files to a working jupyter notebooks.

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

[1]: https://github.com/mwouts/jupytext
[2]: https://gitlab.vgiscience.de/lbsn/tools/jupyterlab