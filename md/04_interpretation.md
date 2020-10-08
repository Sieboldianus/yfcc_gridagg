---
jupyter:
  jupytext:
    formats: notebooks///ipynb,md///md,py///_/py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: worker_env
    language: python
    name: worker_env
---

# Intepretation of HLL data: <br>Comparison, Interactive Exploration, Benchmark Data <a class="tocSkip">

<!-- #region toc=true -->
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Preparations" data-toc-modified-id="Preparations-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Preparations</a></span><ul class="toc-item"><li><span><a href="#Parameters" data-toc-modified-id="Parameters-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Parameters</a></span></li><li><span><a href="#Load-dependencies" data-toc-modified-id="Load-dependencies-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Load dependencies</a></span></li></ul></li><li><span><a href="#Interactive-Map-with-Holoviews/-Geoviews" data-toc-modified-id="Interactive-Map-with-Holoviews/-Geoviews-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Interactive Map with Holoviews/ Geoviews</a></span><ul class="toc-item"><li><span><a href="#Define-base-plotting-functions" data-toc-modified-id="Define-base-plotting-functions-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Define base plotting functions</a></span></li><li><span><a href="#Load-&amp;-plot-pickled-dataframe" data-toc-modified-id="Load-&amp;-plot-pickled-dataframe-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Load &amp; plot pickled dataframe</a></span></li><li><span><a href="#Prepare-interactive-mapping" data-toc-modified-id="Prepare-interactive-mapping-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Prepare interactive mapping</a></span></li><li><span><a href="#Explore-variations" data-toc-modified-id="Explore-variations-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Explore variations</a></span><ul class="toc-item"><li><span><a href="#Use-different-Classification-Scheme" data-toc-modified-id="Use-different-Classification-Scheme-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>Use different Classification Scheme</a></span></li><li><span><a href="#Visualize-under--and-overrepresentation" data-toc-modified-id="Visualize-under--and-overrepresentation-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Visualize under- and overrepresentation</a></span></li></ul></li></ul></li><li><span><a href="#Compare-RAW-and-HLL" data-toc-modified-id="Compare-RAW-and-HLL-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Compare RAW and HLL</a></span></li><li><span><a href="#Working-with-the-Benchmark-data:-Intersection-Example" data-toc-modified-id="Working-with-the-Benchmark-data:-Intersection-Example-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Working with the Benchmark data: Intersection Example</a></span><ul class="toc-item"><li><span><a href="#Load-additional-dependencies:" data-toc-modified-id="Load-additional-dependencies:-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Load additional dependencies:</a></span></li><li><span><a href="#Load-Benchmark-data" data-toc-modified-id="Load-Benchmark-data-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Load Benchmark data</a></span></li><li><span><a href="#Union-hll-sets-for-Countries-UK,-DE-and-FR" data-toc-modified-id="Union-hll-sets-for-Countries-UK,-DE-and-FR-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Union hll sets for Countries UK, DE and FR</a></span><ul class="toc-item"><li><span><a href="#Selection-of-grid-cells-based-on-country-geometry" data-toc-modified-id="Selection-of-grid-cells-based-on-country-geometry-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Selection of grid cells based on country geometry</a></span></li><li><span><a href="#Intersection-with-grid" data-toc-modified-id="Intersection-with-grid-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>Intersection with grid</a></span></li><li><span><a href="#Plot-preview-of-selected-grid-cells-(bins)" data-toc-modified-id="Plot-preview-of-selected-grid-cells-(bins)-4.3.3"><span class="toc-item-num">4.3.3&nbsp;&nbsp;</span>Plot preview of selected grid cells (bins)</a></span></li></ul></li><li><span><a href="#Union-of-hll-sets" data-toc-modified-id="Union-of-hll-sets-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Union of hll sets</a></span></li><li><span><a href="#Calculate-intersection-(common-visitors)" data-toc-modified-id="Calculate-intersection-(common-visitors)-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Calculate intersection (common visitors)</a></span></li><li><span><a href="#Illustrate-intersection-(Venn-diagram)" data-toc-modified-id="Illustrate-intersection-(Venn-diagram)-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Illustrate intersection (Venn diagram)</a></span></li><li><span><a href="#Get-accurate-intersection-counts-(raw)" data-toc-modified-id="Get-accurate-intersection-counts-(raw)-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Get accurate intersection counts (raw)</a></span><ul class="toc-item"><li><span><a href="#Prepare-intersection-with-countries-geometry" data-toc-modified-id="Prepare-intersection-with-countries-geometry-4.7.1"><span class="toc-item-num">4.7.1&nbsp;&nbsp;</span>Prepare intersection with countries geometry</a></span></li><li><span><a href="#Collect-raw-metrics" data-toc-modified-id="Collect-raw-metrics-4.7.2"><span class="toc-item-num">4.7.2&nbsp;&nbsp;</span>Collect raw metrics</a></span></li><li><span><a href="#Update-Venn-Diagram" data-toc-modified-id="Update-Venn-Diagram-4.7.3"><span class="toc-item-num">4.7.3&nbsp;&nbsp;</span>Update Venn Diagram</a></span></li></ul></li><li><span><a href="#Final-plot-(Map-&amp;-Venn-Diagram)" data-toc-modified-id="Final-plot-(Map-&amp;-Venn-Diagram)-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Final plot (Map &amp; Venn Diagram)</a></span></li></ul></li></ul></div>
<!-- #endregion -->

# Introduction

This is the last notebook in a series of four notebooks:

* 1) the [Preparations (01_preparations.ipynb)](01_preparations.html) Basic preparations for processing YFCC100m, explains basic concepts and tools for working with the lbsn data
* 2) the [RAW Notebook (02_yfcc_gridagg_raw.ipynb)](02_yfcc_gridagg_raw.html) demonstrates how a typical grid-based visualization looks like when using the **raw** lbsn structure and  
* 3) the [HLL Notebook (03_yfcc_gridagg_hll.ipynb)](03_yfcc_gridagg_hll.html) demonstrates the same visualization using the privacy-aware **hll** lbsn structure  
* 4) the [Interpretation (04_interpretation.ipynb)](04_interpretation.html) illustrates how to create interactive graphics for comparison of raw and hll results

In this notebook, we'll illustrate how to make further analyses based on the published hll benchmark dataset.

* create interactive map with geoviews, adapt visuals, styling and legend
* combine results from raw and hll into interactive map (on hover)
* store interactive map as standalone HTML
* exporting benchmark data  
* intersecting hll sets for frequentation analysis  


# Preparations 
## Parameters

Load global settings and methods from `03_yfcc_gridagg_hll` notebook. This will also load all parameters and methods defined in `02_yfcc_gridagg_raw`.

```python
import sys
from pathlib import Path
module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)

from _03_yfcc_gridagg_hll import *
```

## Load additional dependencies

Load all dependencies at once, as a means to verify that everything required to run this notebook is available.

```python
import geoviews as gv
import holoviews as hv
import geoviews.feature as gf
import pickle
import ipywidgets as widgets
from ipywidgets.embed import embed_minimal_html
from rtree import index
from geopandas.tools import sjoin
from geoviews import opts
from bokeh.models import HoverTool, FixedTicker
from matplotlib_venn import venn3, venn3_circles
```

Enable bokeh

```python tags=["active-ipynb"]
preparations.init_imports()
```

Activate autoreload of changed python files:

```python tags=["active-ipynb"]
%load_ext autoreload
%autoreload 2
```

Load memory profiler:

```python tags=["active-ipynb"]
%load_ext memory_profiler
```

Print versions of packages used herein, for purposes of replication:

```python tags=["active-ipynb"]
root_packages = [
        'ipywidgets', 'geoviews', 'geopandas', 'holoviews', 'rtree', 'matplotlib-venn', 'xarray']
preparations.package_report(root_packages)
```

# Interactive Map with Holoviews/ Geoviews

Geoviews and Holoviews can be used to create interactive maps, either insider Jupyter or as externally stored as HTML. The syntax of the plot methods must be slightly adapted from the matplotlib output. Given the advanced features of interactivity, we can also add additional information that is shown e.g. on mouse hover over certain grid cells.


## Load & plot pickled dataframe


Loading (geodataframe) using [pickle](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html#pandas.read_pickle). This is the easiest way to load intermediate data, but may be incompatible [if package versions change](https://stackoverflow.com/questions/6687262/how-do-i-know-which-versions-of-pickle-a-particular-version-of-python-supports). If loading pickles does not work, a workaround is to load data from CSV and re-create pickle data, which will be compatible with used versions. Have a look at `03_yfcc_gridagg_hll.ipynb` to recreate pickles with current package versions.


**Load pickle and merge results of raw and hll dataset:**

```python tags=["active-ipynb"]
grid_est = pd.read_pickle(OUTPUT / "pickles" / "yfcc_all_est.pkl")
grid_raw = pd.read_pickle(OUTPUT / "pickles" / "yfcc_all_raw.pkl")
```

```python tags=["active-ipynb"]
grid = grid_est.merge(
    grid_raw[['postcount', 'usercount', 'userdays']],
    left_index=True, right_index=True)
```

Have a look at the numbers for exact and estimated values. Smaller values are exact in both hll and raw because Sparse Mode is used.

```python tags=["active-ipynb"]
grid[grid["usercount_est"]>5].head()
```

## Prepare interactive mapping

Some updates to the plotting function are necessary, for compatibility with geoviews/holoviews interactive mapping.

```python
def label_nodata_xr(
        grid: gp.GeoDataFrame, inverse: bool = None,
        metric: str = "postcount_est", scheme: str = None,
        label_nonesignificant: bool = None, cmap_name: str = None):
    """Create a classified colormap from pandas dataframe
    column with additional No Data-label
        
    Args:
        grid: A geopandas geodataframe with metric column to classify
        inverse: If True, colors are inverted (dark)
        metric: The metric column to classify values
        scheme: The classification scheme to use.
        label_nonesignificant: If True, adds an additional
            color label for none-significant values based
            on column "significant"
        cmap_name: The colormap to use.
            
    Adapted from:
        https://stackoverflow.com/a/58160985/4556479
    See available colormaps:
        http://holoviews.org/user_guide/Colormaps.html
    See available classification schemes:
        https://pysal.org/mapclassify/api.html
    """
    # get headtail_breaks
    # excluding NaN values
    grid_nan = grid[metric].replace(0, np.nan)
    # get classifier scheme by name
    classifier = getattr(mc, scheme)
    # some classification schemes (e.g. HeadTailBreaks)
    # do not support specifying the number of classes returned
    # construct optional kwargs with k == number of classes
    optional_kwargs = {"k":9}
    if scheme == "HeadTailBreaks":
        optional_kwargs = {}
    # explicitly set dtype to float,
    # otherwise mapclassify will error out due
    # to unhashable type numpy ndarray
    # from object-column
    scheme_breaks = classifier(
        grid_nan.dropna().astype(float), **optional_kwargs)
    # set breaks category column
    # to spare cats:
    # increase by 1, 0-cat == NoData/none-significant
    spare_cats = 1
    grid[f'{metric}_cat'] = scheme_breaks.find_bin(
        grid_nan) + spare_cats
    # set cat 0 to NaN values
    # to render NoData-cat as transparent
    grid.loc[grid_nan.isnull(), f'{metric}_cat'] = np.nan
    # get label bounds as flat array
    bounds = get_label_bounds(
        scheme_breaks, grid_nan.dropna().values, flat=True)
    cmap_name = cmap_name
    cmap = plt.cm.get_cmap(cmap_name, scheme_breaks.k)
    # get hex values
    cmap_list = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    # prepend nodata color
    # shown as white in legend
    first_color_legend = '#ffffff'
    if inverse:
        first_color_legend = 'black'
    cmap_list = [first_color_legend] + cmap_list
    cmap_with_nodata = colors.ListedColormap(cmap_list)
    return cmap_with_nodata, bounds, scheme_breaks
```

Since interactive display of grid-polygons is too slow, we're converting the grid to an xarray object, which is then overlayed as a rastered image. Also added here is the option to show additional columns on hover, which will be later used to compare raw and hll values.

```python
def get_custom_tooltips(items: Dict[str, str]) -> str:
    """Compile HoverTool tooltip formatting with items to show on hover"""
    # thousands delimitor formatting
    # will be applied to the for the following columns
    tdelim_format = [
        'usercount_est', 'postcount_est', 'userdays_est',
        'usercount', 'postcount', 'userdays']
    # in HoloViews, custom tooltip formatting can be
    # provided as a list of tuples (name, value)
    tooltips=[
        # f-strings explanation:
        # - k: the item name, v: the item value,
        # - @ means: get value from column
        # optional formatting is applied using
        # `"{,f}" if v in thousand_formats else ""`
        # - {,f} means: add comma as thousand delimiter
        # only for values in tdelim_format (usercount and postcount)
        # else apply automatic format
        (k, 
         f'@{v}{"{,f}" if v.replace("_expected", "") in tdelim_format else ""}'
        ) for k, v in items.items()
    ]
    return tooltips

def set_active_tool(plot, element):
    """Enable wheel_zoom in bokeh plot by default"""
    plot.state.toolbar.active_scroll = plot.state.tools[0]

def convert_gdf_to_gvimage(
        grid, metric, cat_count,
        additional_items: Dict[str, str] = None) -> gv.Image:
    """Convert GeoDataFrame to gv.Image using categorized
    metric column as value dimension
    
    Args:
        grid: A geopandas geodataframe with indexes x and y 
            (projected coordinates) and aggregate metric column
        metric: target column for value dimension.
            "_cat" will be added to retrieve classified values.
        cat_count: number of classes for value dimension
        additional_items: a dictionary with optional names 
            and column references that are included in 
            gv.Image to provide additional information
            (e.g. on hover)
    """
    if additional_items is None:
        additional_items_list = []
    else:
        additional_items_list = [
            v for v in additional_items.values()]
    # convert GeoDataFrame to xarray object
    # the first vdim is the value being used 
    # to visualize classes on the map
    # include additional_items (postcount and usercount)
    # to show exact information through tooltip
    xa_dataset = gv.Dataset(
        grid.to_xarray(),
        vdims=[
            hv.Dimension(
                f'{metric}_cat', range=(0, cat_count))]
            + additional_items_list,
        crs=crs.Mollweide())
    return xa_dataset.to(gv.Image, crs=crs.Mollweide())
    
def plot_interactive(grid: gp.GeoDataFrame, title: str,
    metric: str = "postcount_est", store_html: str = None,
    additional_items: Dict[str, str] = {
        'Post Count':'postcount',
        'User Count':'usercount',
        'User Days':'userdays',
        'Post Count (estimated)':'postcount_est', 
        'User Count (estimated)':'usercount_est',
        'User Days (estimated)':'userdays_est'},
    scheme: str = "HeadTailBreaks",
    cmap: str = "OrRd",
    inverse: bool = None,
    output: Path = OUTPUT):
    """Plot interactive map with holoviews/geoviews renderer

    Args:
        grid: A geopandas geodataframe with indexes x and y 
            (projected coordinates) and aggregate metric column
        metric: target column for aggregate. Default: postcount.
        store_html: Provide a name to store figure as interactive HTML.
        title: Title of the map
        additional_items: additional items to show on hover
        scheme: The classification scheme to use. Default "HeadTailBreaks".
        cmap: The colormap to use. Default "OrRd".
        inverse: plot other map elements inverse (black)
        output: main folder (Path) for storing output
    """
    # check if all additional items are available
    for key, item in list(additional_items.items()):
        if item not in grid.columns:
            additional_items.pop(key)
    # work on a shallow copy to not modify original dataframe
    grid_plot = grid.copy()
    # classify metric column
    cmap_with_nodata, bounds, headtail_breaks = label_nodata_xr(
        grid=grid_plot, inverse=inverse, metric=metric,
        scheme=scheme, cmap_name=cmap)
    # construct legend labels for colormap
    label_dict = {}
    for i, s in enumerate(bounds):
        label_dict[i+1] = s
    label_dict[0] = "No data"
    cat_count = len(bounds)
    # create gv.image layer from gdf
    img_grid = convert_gdf_to_gvimage(
            grid=grid_plot,
            metric=metric, cat_count=cat_count,
            additional_items=additional_items)
    # define additional plotting parameters
    # width of static jupyter map,
    # 360° == 1200px
    width = 1200 
    # height of static jupyter map,
    # 360°/2 == 180° == 600px
    height = int(width/2) 
    responsive = False
    aspect = None
    # if stored as html,
    # override values
    if store_html:
        width = None
        height = None
        responsive = True
    # define width and height as optional parameters
    # only used when plotting inside jupyter
    optional_kwargs = dict(width=width, height=height)
    # compile only values that are not None into kwargs-dict
    # by using dict-comprehension
    optional_kwargs_unpack = {
        k: v for k, v in optional_kwargs.items() if v is not None}
    # prepare custom HoverTool
    tooltips = get_custom_tooltips(
        additional_items)
    hover = HoverTool(tooltips=tooltips) 
    # create image layer
    image_layer = img_grid.opts(
            color_levels=cat_count,
            cmap=cmap_with_nodata,
            colorbar=True,
            clipping_colors={'NaN': 'transparent'},
            colorbar_opts={
                # 'formatter': formatter,
                'major_label_text_align':'left',
                'major_label_overrides': label_dict,
                'ticker': FixedTicker(
                    ticks=list(range(0, len(label_dict)))),
                },
            tools=[hover],
            # optional unpack of width and height
            **optional_kwargs_unpack
        )
    edgecolor = 'black'
    bg_color = None
    fill_color = '#479AD4'
    alpha = 0.1
    if inverse:
        alpha = 1
        bg_color = 'black'
        edgecolor = 'white'
    # combine layers into single overlay
    # and set global plot options
    gv_layers = (gf.ocean.opts(alpha=alpha, fill_color=fill_color) * \
                 image_layer * \
                 gf.coastline.opts(line_color=edgecolor) * \
                 gf.borders.opts(line_color=edgecolor)
            ).opts(
        bgcolor=bg_color,
        # global_extent=True,
        projection=crs.Mollweide(),
        responsive=responsive,
        data_aspect=1, # maintain fixed aspect ratio during responsive resize
        hooks=[set_active_tool],
        title=title)
    if store_html:
        hv.save(gv_layers, output / "html" / f'{store_html}.html', backend='bokeh')
    else:
        return gv_layers
```

**plot interactive in-line:**

```python tags=["active-ipynb"]
gv_plot = plot_interactive(
    grid, title=f'YFCC User Count (estimated) per {int(GRID_SIZE_METERS/1000):.0f}km grid', metric="usercount_est",)
gv_plot
```

**store as html:**

```python tags=["active-ipynb"]
plot_interactive(
    grid, title=f'YFCC User Count (estimated) per {int(GRID_SIZE_METERS/1000):.0f}km grid', metric="usercount_est",
    store_html="yfcc_usercount_est")
```

**Interactive map**


View the interactive map [here](../out/html/yfcc_usercount_est.html) and hover over bins to compare exact (raw) vs. approximate (hll) results.


# Compare RAW and HLL

We're almost at the end of the end of the three-part tutorial. The last step is to compare and interpret results.


**Post Count, User Count and User Days**


Load png plots as [widgets.Image](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#Image) and display using [widgets.Tab](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#Tabs)

```python tags=["active-ipynb"]
# dictionary with filename and title
pathrefs = {
    0: ('yfcc_postcount.png', 'Post Count raw'),
    1: ('yfcc_postcount_est.png', 'Post Count hll'),
    2: ('yfcc_usercount.png', 'User Count raw'),
    3: ('yfcc_usercount_est.png', 'User Count hll'),
    4: ('yfcc_userdays.png', 'User Days raw'),
    5: ('yfcc_userdays_est.png', 'User Days hll'),
    6: ('postcount_sample.png', 'Post Count (Italy) raw'),
    7: ('postcount_sample_est.png', 'Post Count (Italy) hll'),
    8: ('usercount_sample.png', 'User Count (Italy) raw'),
    9: ('usercount_sample_est.png', 'User Count (Italy) hll'),
    10: ('userdays_sample.png', 'User Days (Italy) raw'),
    11: ('userdays_sample_est.png', 'User Days (Italy) hll'),
    }

def get_img_width(filename: str):
    if 'sample' in filename:
        return 700
    return 1300

widgets_images = [
    widgets.Image(
        value=open(OUTPUT / "figures" / pathref[0], "rb").read(),
        format='png',
        width=get_img_width(pathref[0])
     )
    for pathref in pathrefs.values()]
```

Configure tabs

```python tags=["active-ipynb"]
children = widgets_images
tab = widgets.Tab()
tab.children = children
for i in range(len(children)):
    tab.set_title(i, pathrefs[i][1])
```

Display inside live notebook:

```python tags=["active-ipynb"]
tab
```

The above tab display [is not available](https://ipywidgets.readthedocs.io/en/latest/embedding.html) in the static notebook HTML export. A standalone HTML with the above tabs can be generated with the following command:

```python tags=["active-ipynb"]
embed_minimal_html(
    OUTPUT / 'html' / 'yfcc_compare_raw_hll.html',
    views=[tab], title='YFCC worldwide raw and hll metrics')
```

View the result [here](../out/html/yfcc_compare_raw_hll.html).


# Working with the Benchmark data: Intersection Example

HLL sets are more than just statistic summaries. Except for removing elements, they offer the regular set operations such as lossless union. Based on the [Inclusion–exclusion principle](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle), Unions of two or more sets can be used to calculate the intersection, i.e. the common elements between all sets. In the following, intersection is calculated for common users between France, Germany, and UK.


## Load Benchmark data

In [03_yfcc_gridagg_hll.ipynb](03_yfcc_gridagg_hll.ipynb), aggregate grid data was stored to [yfcc_all_est_benchmark.csv](yfcc_all_est_benchmark.csv), including hll sets with usercount cardinality > 100. Load this data first, using functions from previous notebooks.


Read benchmark data, only loading usercount and usercount_hll columns.

```python tags=["active-ipynb"]
grid = grid_agg_fromcsv(
    OUTPUT / "csv" / "yfcc_all_est_benchmark.csv",
    columns=["xbin", "ybin", "usercount_hll"],
    metrics=["usercount_est"],
    grid_size=GRID_SIZE_METERS)
```

```python tags=["active-ipynb"]
grid[grid["usercount_est"]>5].head()
```

## Union hll sets for Countries UK, DE and FR
### Selection of grid cells based on country geometry

Load country geometry:

```python tags=["active-ipynb"]
world = gp.read_file(
    gp.datasets.get_path('naturalearth_lowres'),
    crs=CRS_WGS)
world = world.to_crs(CRS_PROJ)
```

Select geometry for DE, FR and UK

```python tags=["active-ipynb"]
de = world[world['name'] == "Germany"]
uk = world[world['name'] == "United Kingdom"]
fr = world[world['name'] == "France"]
```

Drop French territory of French Guiana:

```python tags=["active-ipynb"]
fr = fr.explode().iloc[1:].dissolve(by='name')
fr.plot()
```

Preview selection. Note that the territory of France includes Corsica, which is acceptable for the example use case.

```python tags=["active-ipynb"]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle(
    'Areas to test for common visitors in the hll benchmark dataset')
for ax in (ax1, ax2, ax3):
    ax.set_axis_off()
ax1.title.set_text('DE')
ax2.title.set_text('UK')
ax3.title.set_text('FR')
de.plot(ax=ax1)
uk.plot(ax=ax2)
fr.plot(ax=ax3)
```

### Intersection with grid

Since grid size is 100 km, direct intersection will yield some error rate. Use centroid of grid cells to select bins based on country geometry.

Get centroids as Geoseries and turn into GeoDataFrame:

```python tags=["active-ipynb"]
centroid_grid = grid.centroid.reset_index()
centroid_grid.set_index(["xbin", "ybin"], inplace=True)
```

```python tags=["active-ipynb"]
grid.centroid
```

Define function to intersection, using geopandas [sjoin (spatial join)](https://geopandas.org/reference/geopandas.sjoin.html)

```python
def intersect_grid_centroids(
    grid: gp.GeoDataFrame, 
    intersect_gdf: gp.GeoDataFrame):
    """Return grid centroids from grid that 
    intersect with intersect_gdf
    """
    centroid_grid = gp.GeoDataFrame(
        grid.centroid)
    centroid_grid.rename(
        columns={0:'geometry'},
        inplace=True)
    centroid_grid.set_geometry(
        'geometry', crs=grid.crs, 
        inplace=True)
    grid_intersect = sjoin(
        centroid_grid, intersect_gdf, 
        how='right')
    grid_intersect.set_index(
        ["index_left0", "index_left1"],
        inplace=True)
    grid_intersect.index.names = ['xbin','ybin']
    return grid.loc[grid_intersect.index]
```

Run intersection for countries:

```python tags=["active-ipynb"]
grid_de = intersect_grid_centroids(
    grid=grid, intersect_gdf=de)
grid_de.plot(edgecolor='white')
```

```python tags=["active-ipynb"]
grid_fr = intersect_grid_centroids(
    grid=grid, intersect_gdf=fr)
grid_fr.plot(edgecolor='white')
```

```python tags=["active-ipynb"]
grid_uk = intersect_grid_centroids(
    grid=grid, intersect_gdf=uk)
grid_uk.plot(edgecolor='white')
```

### Plot preview of selected grid cells (bins)

Define colors:

```python tags=["active-ipynb"]
color_de = "#fc4f30"
color_fr = "#008fd5"
color_uk = "#6d904f"
```

Define map boundary:

```python tags=["active-ipynb"]
bbox_europe = (
    -9.580078, 41.571384,
    16.611328, 59.714117)
minx, miny = PROJ_TRANSFORMER.transform(
    bbox_europe[0], bbox_europe[1])
maxx, maxy = PROJ_TRANSFORMER.transform(
    bbox_europe[2], bbox_europe[3])
buf = 100000
```

```python
def plot_map(
    grid: gp.GeoDataFrame, sel_grids: List[gp.GeoDataFrame],
    sel_colors: List[str],
    title: Optional[str] = None, save_fig: Optional[str] = None,
    ax = None):
    """Plot GeoDataFrame with matplotlib backend, optionaly export as png"""
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    ax.set_xlim(minx-buf, maxx+buf)
    ax.set_ylim(miny-buf, maxy+buf)
    if title:
        ax.set_title(title, fontsize=12)
    for ix, sel_grid in enumerate(sel_grids):
        sel_grid.plot(
            ax=ax,
            color=sel_colors[ix],
            edgecolor='white',
            alpha=0.9)
    grid.boundary.plot(
        ax=ax,
        edgecolor='black', 
        linewidth=0.1,
        alpha=0.9)
    # combine with world geometry
    world.plot(
        ax=ax, color='none', edgecolor='black', linewidth=0.3)
    # turn axis off
    ax.set_axis_off()
    if not save_fig:
        return
    fig.savefig(OUTPUT / "figures" / save_fig, dpi=300, format='PNG',
               bbox_inches='tight', pad_inches=1)
```

```python tags=["active-ipynb"]
sel_grids=[grid_de, grid_uk, grid_fr]
sel_colors=[color_de, color_uk, color_fr]
plot_map(
    grid=grid, sel_grids=sel_grids, 
    sel_colors=sel_colors,
    title='Grid selection for DE, FR and UK',
    save_fig='grid_selection_countries.png')
```

## Union of hll sets


Connect to hll worker:

```python tags=["active-ipynb"]
db_user = "postgres"
db_pass = os.getenv('POSTGRES_PASSWORD')
# set connection variables
db_host = "hlldb"
db_port = "5432"
db_name = "hlldb"

db_connection = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass
)
db_connection.set_session(readonly=True)
db_conn = tools.DbConn(db_connection)
db_conn.query("SELECT 1;")
```

Use `union_all` function from `03_yfcc_gridagg_hll.ipynb`, but modify to union all hll sets in a pd.Series (`union_all`), instead of grouped union.

```python
def union_all_hll(
    hll_series: pd.Series, cardinality: bool = True) -> pd.Series:
    """HLL Union and (optional) cardinality estimation from series of hll sets

        Args:
        hll_series: Indexed series (bins) of hll sets. 
        cardinality: If True, returns cardinality (counts). Otherwise,
            the unioned hll set will be returned.
    """
    hll_values_list = ",".join(
        [f"(0::int,'{hll_item}'::hll)"
         for hll_item in hll_series.values.tolist()])
    return_col = "hll_union"
    hll_calc_pre = ""
    hll_calc_tail = "AS hll_union"
    if cardinality:
        return_col = "hll_cardinality"
        hll_calc_pre = "hll_cardinality("
        hll_calc_tail = ")::int"
    db_query = f"""
        SELECT sq.{return_col} FROM (
            SELECT s.group_ix,
                   {hll_calc_pre}
                   hll_union_agg(s.hll_set)
                   {hll_calc_tail}
            FROM (
                VALUES {hll_values_list}
                ) s(group_ix, hll_set)
            GROUP BY group_ix
            ORDER BY group_ix ASC) sq
        """
    df = db_conn.query(db_query)
    return df[return_col]
```

Calculate distinct users per country:

```python tags=["active-ipynb"]
grid_sel = {
    "de": grid_de,
    "uk": grid_uk,
    "fr": grid_fr
}
distinct_users_total = {}
for country, grid_sel in grid_sel.items():
    # drop bins with no values
    cardinality_series = union_all_hll(
        grid_sel["usercount_hll"].dropna())
    distinct_users_total[country] = cardinality_series[0]
    print(
        f"{distinct_users_total[country]} distinct users "
        f"who shared YFCC100M photos in {country.upper()}")
```

## Calculate intersection (common visitors)

According to the [Union-intersection-principle](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle):

$|A \cup B| = |A| + |B| - |A \cap B|$

which can also be written as:

$|A \cap B| = |A| + |B| - |A \cup B|$

Therefore, unions can be used to calculate intersection. Calculate $|DE \cup FR|$, $|DE \cup UK|$ and $|UK \cup FR|$, i.e.:
```
hll_cardinality(grid_de)::int + 
hll_cardinality(grid_fr)::int - 
hll_cardinality(hll_union(grid_de, grid_fr') = IntersectionCount
```

```python tags=["active-ipynb"]
union_de_fr = pd.concat([grid_de, grid_fr])
union_de_uk = pd.concat([grid_de, grid_uk])
union_uk_fr = pd.concat([grid_uk, grid_fr])
```

```python tags=["active-ipynb"]
grid_sel = {
    "de-uk": union_de_uk,
    "de-fr": union_de_fr,
    "uk-fr": union_uk_fr
}
distinct_common = {}
for country_tuple, grid_sel in grid_sel.items():
    cardinality_series = union_all_hll(
        grid_sel["usercount_hll"].dropna())
    distinct_common[country_tuple] = cardinality_series[0]
    print(
        f"{distinct_common[country_tuple]} distinct total users "
        f"who shared YFCC100M photos from either {country_tuple.split('-')[0]} "
        f"or {country_tuple.split('-')[1]} (union)")
```

Calculate intersection:

```python tags=["active-ipynb"]
distinct_intersection = {}
for a, b in [("de", "uk"), ("de", "fr"), ("uk", "fr")]:
    a_total = distinct_users_total[a]
    b_total = distinct_users_total[b]
    common_ref = f'{a}-{b}'
    intersection_count = a_total + b_total - distinct_common[common_ref]
    distinct_intersection[common_ref] = intersection_count
    print(
        f"{distinct_intersection[common_ref]} distinct users "
        f"who shared YFCC100M photos from {a} and {b} (intersection)")
```

Finally, lets get the number of users who have shared pictures from all three countries, based on the [formula for three sets](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle):

$|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|$

which can also be written as:

$|A \cap B \cap C| = |A \cup B \cup C| - |A| - |B| - |C| + |A \cap B| + |A \cap C| + |B \cap C|$


Calculate distinct users of all three countries:

```python tags=["active-ipynb"]
union_de_fr_uk = pd.concat(
    [grid_de, grid_fr, grid_uk])
cardinality_series = union_all_hll(
    union_de_fr_uk["usercount_hll"].dropna())
union_count_all = cardinality_series[0]
union_count_all
```

```python tags=["active-ipynb"]
country_a = "de"
country_b = "uk"
country_c = "fr"
```

Calculate intersection:

```python tags=["active-ipynb"]
intersection_count_all = union_count_all - \
    distinct_users_total[country_a] - \
    distinct_users_total[country_b] - \
    distinct_users_total[country_c] + \
    distinct_intersection[f'{country_a}-{country_b}'] + \
    distinct_intersection[f'{country_a}-{country_c}'] + \
    distinct_intersection[f'{country_b}-{country_c}']
    
print(intersection_count_all)
```

Since we're going to visualize this with [matplotlib-venn](https://github.com/konstantint/matplotlib-venn),
we need the following variables:

```python tags=["active-ipynb"]
v = venn3(
    subsets=(
        500,
        500, 
        100,
        500,
        100,
        100,
        10),
    set_labels = ('A', 'B', 'C'))
v.get_label_by_id('100').set_text('Abc')
v.get_label_by_id('010').set_text('aBc')
v.get_label_by_id('001').set_text('abC')
v.get_label_by_id('110').set_text('ABc')
v.get_label_by_id('101').set_text('AbC')
v.get_label_by_id('011').set_text('aBC')
v.get_label_by_id('111').set_text('ABC')
plt.show()
```

We already have `ABC`, the other values can be calulated:

```python tags=["active-ipynb"]
ABC = intersection_count_all
```

```python tags=["active-ipynb"]
ABc = distinct_intersection[f'{country_a}-{country_b}'] - ABC
```

```python tags=["active-ipynb"]
aBC = distinct_intersection[f'{country_b}-{country_c}'] - ABC
```

```python tags=["active-ipynb"]
AbC = distinct_intersection[f'{country_a}-{country_c}'] - ABC
```

```python tags=["active-ipynb"]
Abc = distinct_users_total[country_a] - ABc - AbC + ABC
```

```python tags=["active-ipynb"]
aBc = distinct_users_total[country_b] - ABc - aBC + ABC
```

```python tags=["active-ipynb"]
abC = distinct_users_total[country_c] - aBC - AbC + ABC
```

## Illustrate intersection (Venn diagram)

Order of values handed over: Abc, aBc, ABc, abC, AbC, aBC, ABC

Define Function to plot Venn Diagram.

```python
def plot_venn(
    subset_sizes: List[int],
    colors: List[str], 
    names: List[str],
    subset_sizes_raw: List[int] = None,
    total_sizes: List[Tuple[int, int]] = None,
    ax = None,
    title: str = None):
    """Plot Venn Diagram"""
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
    set_labels = (
        'A', 'B', 'C')
    v = venn3(
        subsets=(
            [subset_size for subset_size in subset_sizes]),
        set_labels = set_labels,
        ax=ax)    
    for ix, idx in enumerate(
        ['100', '010', '001']):
        v.get_patch_by_id(
            idx).set_color(colors[ix])
        v.get_patch_by_id(
            idx).set_alpha(0.8)
        v.get_label_by_id(
            set_labels[ix]).set_text(
            names[ix])
        if not total_sizes:
            continue
        raw_count = total_sizes[ix][0]
        hll_count = total_sizes[ix][1]
        difference = abs(raw_count-hll_count)
        v.get_label_by_id(set_labels[ix]).set_text(
            f'{names[ix]}, {hll_count},\n'
            f'{difference/(raw_count/100):+.1f}%')
    if subset_sizes_raw:
        for ix, idx in enumerate(
            ['100', '010', None, '001']):
            if not idx:
                continue
            dif_abs = subset_sizes[ix] - subset_sizes_raw[ix]
            dif_perc = dif_abs / (subset_sizes_raw[ix] / 100)
            v.get_label_by_id(idx).set_text(
                f'{subset_sizes[ix]}\n{dif_perc:+.1f}%')            
    label_ids = [
        '100', '010', '001',
        '110', '101', '011',
        '111', 'A', 'B', 'C']
    for label_id in label_ids:
        v.get_label_by_id(
            label_id).set_fontsize(14)
    # draw borders
    c = venn3_circles(
        subsets=(
            [subset_size for subset_size in subset_sizes]),
        linestyle='dashed',
        lw=1,
        ax=ax)
    if title:
        ax.title.set_text(title)
```

Plot Venn Diagram:

```python tags=["active-ipynb"]
subset_sizes = [
    Abc, aBc, ABc, abC, AbC, aBC, ABC]
colors = [
    color_de, color_uk, color_fr]
names = [
    'Germany', 'United Kingdom','France']
plot_venn(
    subset_sizes=subset_sizes,
    colors=colors,
    names=names,
    title="Common User Count")
```

**Combine Map & Venn Diagram**

```python tags=["active-ipynb"]
# figure with subplot (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize=(10, 24))
plot_map(
    grid=grid, sel_grids=sel_grids, 
    sel_colors=sel_colors, ax=ax[0])
plot_venn(
    subset_sizes=subset_sizes,
    colors=colors,
    names=names,
    ax=ax[1])
# store as png
fig.savefig(
    OUTPUT / "figures" / "figure_2_intersection.png", dpi=300, format='PNG',
    bbox_inches='tight', pad_inches=1)
```

## Get accurate intersection counts (raw)

We're going to work on the raw post data that was stored from raw db using GeoHash accuracy of 5 (~4km Granularity).

```python tags=["active-ipynb"]
usecols = ['latitude', 'longitude', 'user_guid']
dtypes = {'latitude': float, 'longitude': float, 'user_guid': str}
```

Convert Europe bounding box to Mollweide and get buffer

```python tags=["active-ipynb"]
minx, miny = PROJ_TRANSFORMER.transform(
    bbox_europe[0], bbox_europe[1])
maxx, maxy = PROJ_TRANSFORMER.transform(
    bbox_europe[2], bbox_europe[3])
# apply buffer and convetr back to WGS1984
min_buf = PROJ_TRANSFORMER_BACK.transform(minx-buf, miny-buf)
max_buf = PROJ_TRANSFORMER_BACK.transform(maxx+buf, maxy+buf)
bbox_europe_buf = (min_buf[0], min_buf[1], max_buf[0], max_buf[1])
```

```python tags=["active-ipynb"]
%%time
int_data = 'user_guids_sets_raw.pkl'
skip_read_from_raw = False
if (OUTPUT / "pickles" / int_data).exists():
    print(
        f"Intermediate data already exists. "
        f"Delete {int_data} to reload from raw.")
    skip_read_from_raw = True
else:
    chunked_df = read_project_chunked(
        filename=OUTPUT / "csv" / "yfcc_posts.csv",
        usecols=usecols,
        bbox=bbox_europe_buf,
        dtypes=dtypes)
    print(len(chunked_df))
    display(chunked_df[0].head())
    display(chunked_df[0]["x"])
```

### Prepare intersection with countries geometry

Explode Countries Geometry and join in single GeoDataFrame. 

A bug in [geopandas](https://github.com/geopandas/geopandas/pull/1319/commits/f6dfe9fdf8524d7b8f1e94c48a6fa149a93ce3f0) prevents to use `explode()` on France, but we can use the following workaround:

```python tags=["active-ipynb"]
exploded_geom = fr.geometry.explode().reset_index(level=-1)
exploded_index = exploded_geom.columns[0]
fr_exploded = fr.drop(fr._geometry_column_name, axis=1).join(exploded_geom)
```

```python tags=["active-ipynb"]
de_uk_fr = pd.concat(
    [de.explode(), uk.explode(), fr_exploded.reset_index()]).reset_index()
de_uk_fr.drop(
    de_uk_fr.columns.difference(['name','geometry']), 1, inplace=True)
```

```python tags=["active-ipynb"]
de_uk_fr.head()
```

Populate R-tree index with bounds of polygons

```python tags=["active-ipynb"]
idx = index.Index()
for pos, poly in enumerate(de_uk_fr.geometry):
    idx.insert(pos, poly.bounds)
```

Test with single point

```python tags=["active-ipynb"]
coords = (10, 50)
coords = PROJ_TRANSFORMER.transform(
    coords[0], coords[1])
for ix in idx.intersection(coords):
    if de_uk_fr.geometry[ix].contains(Point(coords)):
        print(de_uk_fr.name[ix])
de_uk_fr.geometry[ix]
```

Query post locations to see which polygon it is in
using first Rtree index, then Shapely geometry's within

```python
USER_GUIDS_SETS = {
    "Germany": set(),
    "United Kingdom": set(),
    "France": set()
}

def intersect_coord(
    coordinates: Tuple[float, float],
    gdf_country: gp.GeoDataFrame,
    user_guid: str):
    """Intersect coordinates with GeoDataFrame geometry"""
    # rtree bounds intersection
    for ix in idx.intersection(coordinates):
        # exact shape intersection
        if not gdf_country.geometry[ix].contains(
            Point(coordinates)):
            continue
        country = gdf_country.name[ix]
        USER_GUIDS_SETS[country].add(
            user_guid)
```

This is likely not the most performant solution, but it is easy to read:

```python
def apply_intersect_coord(
    gdf_country: gp.GeoDataFrame,
    chunked_df: List[pd.DataFrame]):
    """Assign user guids from posts from chunked_df to countries
    and union in global sets
    """
    processed_cnt = 0
    for df in chunked_df:
        for post in df.itertuples():
            processed_cnt += 1
            intersect_coord(
                coordinates=(post.x, post.y),
                gdf_country=gdf_country,
                user_guid=post.user_guid)
            # report
            if processed_cnt % 1000 == 0:
                print(f'Processed {processed_cnt:,.0f} posts')
                clear_output(wait=True)
```

Run:

```python tags=["active-ipynb"]
%%time
%%memit
if not skip_read_from_raw:
    apply_intersect_coord(
        chunked_df=chunked_df,
        gdf_country=de_uk_fr)
```

**Store intermediate data**

```python tags=["active-ipynb"]
if not skip_read_from_raw:
    with open(OUTPUT / 'pickles' / 'user_guids_sets_raw.pkl', 'wb') as handle:
        pickle.dump(
            USER_GUIDS_SETS,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL)
```

Size:

```python tags=["active-ipynb"]
user_guids_raw = Path(OUTPUT / 'pickles' / int_data).stat().st_size / (1024*1024)
print(f"Size: {user_guids_raw:.2f} MB")
```

Load intermediate data:

```python tags=["active-ipynb"]
with open(OUTPUT / 'pickles' / int_data, 'rb') as handle:
    USER_GUIDS_SETS = pickle.load(handle)
```

Output total number of distinct users per country:

```python tags=["active-ipynb"]
for ix, (country, user_guids_set) in enumerate(USER_GUIDS_SETS.items()):
    raw_count = len(user_guids_set)
    hll_count = list(distinct_users_total.values())[ix]
    difference = abs(raw_count-hll_count)
    print(
        f'{raw_count} total users in {country} '
        f'({hll_count} estimated hll, '
        f'{difference/(raw_count/100):.02f}% total error)')
```

<!-- #region -->
### Collect raw metrics

Apply python [set operations](https://docs.python.org/3.8/library/stdtypes.html#set):
```python
A.union(B)
A | B | C
```
```python
A.intersection(B)
A & B & C
```
```python
A.difference(B)
A - B - C
```
<!-- #endregion -->

```python tags=["active-ipynb"]
ABC_raw = len(USER_GUIDS_SETS["Germany"] & USER_GUIDS_SETS["United Kingdom"] & USER_GUIDS_SETS["France"])
ABC_raw
```

```python tags=["active-ipynb"]
ABc_raw = len(USER_GUIDS_SETS["Germany"] & USER_GUIDS_SETS["United Kingdom"]) - ABC_raw
ABc_raw
```

```python tags=["active-ipynb"]
aBC_raw = len(USER_GUIDS_SETS["United Kingdom"] & USER_GUIDS_SETS["France"]) - ABC_raw
aBC_raw
```

```python tags=["active-ipynb"]
AbC_raw = len(USER_GUIDS_SETS["Germany"] & USER_GUIDS_SETS["France"]) - ABC_raw
AbC_raw
```

```python tags=["active-ipynb"]
Abc_raw = len(USER_GUIDS_SETS["Germany"]) - ABc_raw - AbC_raw + ABC_raw
Abc_raw
```

```python tags=["active-ipynb"]
aBc_raw = len(USER_GUIDS_SETS["United Kingdom"]) - ABc_raw - aBC_raw + ABC_raw
aBc_raw
```

```python tags=["active-ipynb"]
abC_raw = len(USER_GUIDS_SETS["France"]) - aBC_raw - AbC_raw + ABC_raw
abC_raw
```

### Update Venn Diagram

With raw measurements and error rate for hll measurements

```python tags=["active-ipynb"]
# figure with subplot (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize=(10, 30))
subset_sizes_raw = [
    Abc_raw, aBc_raw, ABc_raw, abC_raw, AbC_raw, aBC_raw, ABC_raw]
colors = [
    color_de, color_uk, color_fr]
names = [
    'Germany', 'United Kingdom','France']
plot_venn(
    subset_sizes=subset_sizes_raw,
    colors=colors,
    names=names,
    title="Common User Count (Raw)",
    ax=ax[0])
plot_venn(
    subset_sizes=subset_sizes,
    colors=colors,
    names=names,
    title="Common User Count (Hll)",
    ax=ax[1])
# adjust horizontal spacing
fig.tight_layout(pad=3.0)
```

## Final plot (Map & Venn Diagram)

For the final plot, add raw counts and error rates as additional annotation to Venn Diagram.

```python tags=["active-ipynb"]
# figure with subplot (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize=(10, 24))
names = [
    'DE', 'UK','FR']  
total_sizes = [
    (len(user_guids_set), list(distinct_users_total.values())[ix])
    for ix, user_guids_set in enumerate(USER_GUIDS_SETS.values())
    ]
plot_map(
    grid=grid, sel_grids=sel_grids, 
    sel_colors=sel_colors,
    ax=ax[0])
plot_venn(
    subset_sizes=subset_sizes,
    subset_sizes_raw=subset_sizes_raw,
    total_sizes=total_sizes,
    colors=colors,
    names=names,
    ax=ax[1])
# add annotations for inner error labels
# this is fuzzy work
label_pos = {
    '110': (10, 50),
    '101': (-90, -50),
    '011': (90, -50),
    '111': (-110, 10),
}
label_rad = {
    '110': 0.1,
    '101': 0.5,
    '011': 0.3,
    '111': -0.1,
}
arr_off = {
    '110': [0, -0.05],
    '101': [0.05, 0.05],
    '011': [-0.07, 0],
    '111': [0.15, 0],
}
for ix, label_id in enumerate(
    [None, None, '110', None, '101', '011', '111']):
    if not label_id:
        continue
    dif_abs = subset_sizes[ix] - subset_sizes_raw[ix]
    dif_perc = dif_abs / (subset_sizes_raw[ix] / 100)
    label = f'{dif_perc:+.1f}%'
    
    ax[1].annotate(
        label,
        xy=v.get_label_by_id(
            label_id).get_position() - np.array(arr_off[label_id]),
        xytext=label_pos[label_id],
        ha='center',
        textcoords='offset points',
        bbox=dict(
            boxstyle='round,pad=0.5',
            fc='gray',
            alpha=0.1),
        arrowprops=dict(
            arrowstyle='->', 
            connectionstyle=f'arc3,rad={label_rad[label_id]}',
            color='gray'))
# store as png
fig.savefig(
    OUTPUT / "figures" / "figure_2_intersection.png", dpi=300, format='PNG',
    bbox_inches='tight', pad_inches=1)
plt.show()
```

# Close DB connection & Create notebook HTML

```python tags=["active-ipynb"]
db_connection.close ()
```

```python tags=["active-ipynb"]
!jupyter nbconvert --to html_toc \
    --output-dir=../out/html ./04_interpretation.ipynb \
    --template=../nbconvert.tpl \
    --ExtractOutputPreprocessor.enabled=False # create single output file
```

```python

```

```python

```
