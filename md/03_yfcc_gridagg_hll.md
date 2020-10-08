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

# Privacy-aware YFCC100m visualization based on 100x100km grid (Mollweide) <a class="tocSkip">

<!-- #region toc=true -->
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Preparations" data-toc-modified-id="Preparations-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Preparations</a></span><ul class="toc-item"><li><span><a href="#Load-dependencies" data-toc-modified-id="Load-dependencies-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Load dependencies</a></span></li><li><span><a href="#Parameters" data-toc-modified-id="Parameters-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Parameters</a></span><ul class="toc-item"><li><span><a href="#Connect-to-database" data-toc-modified-id="Connect-to-database-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Connect to database</a></span></li></ul></li><li><span><a href="#Get-data-from-db-and-write-to-CSV" data-toc-modified-id="Get-data-from-db-and-write-to-CSV-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Get data from db and write to CSV</a></span></li><li><span><a href="#Create-Grid" data-toc-modified-id="Create-Grid-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Create Grid</a></span></li><li><span><a href="#Prepare-functions" data-toc-modified-id="Prepare-functions-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Prepare functions</a></span></li><li><span><a href="#Test-with-LBSN-data" data-toc-modified-id="Test-with-LBSN-data-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Test with LBSN data</a></span><ul class="toc-item"><li><span><a href="#Load-data" data-toc-modified-id="Load-data-2.6.1"><span class="toc-item-num">2.6.1&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Project-coordinates-to-Mollweide" data-toc-modified-id="Project-coordinates-to-Mollweide-2.6.2"><span class="toc-item-num">2.6.2&nbsp;&nbsp;</span>Project coordinates to Mollweide</a></span></li><li><span><a href="#Perform-the-bin-assignment" data-toc-modified-id="Perform-the-bin-assignment-2.6.3"><span class="toc-item-num">2.6.3&nbsp;&nbsp;</span>Perform the bin assignment</a></span></li></ul></li><li><span><a href="#A:-Estimated-Post-Count-per-grid" data-toc-modified-id="A:-Estimated-Post-Count-per-grid-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>A: Estimated Post Count per grid</a></span><ul class="toc-item"><li><span><a href="#Preview-post-count-map" data-toc-modified-id="Preview-post-count-map-2.7.1"><span class="toc-item-num">2.7.1&nbsp;&nbsp;</span>Preview post count map</a></span></li></ul></li><li><span><a href="#B:-Estimated-User-Count-per-grid" data-toc-modified-id="B:-Estimated-User-Count-per-grid-2.8"><span class="toc-item-num">2.8&nbsp;&nbsp;</span>B: Estimated User Count per grid</a></span><ul class="toc-item"><li><span><a href="#Perform-the-bin-assignment-and-estimate-distinct-users" data-toc-modified-id="Perform-the-bin-assignment-and-estimate-distinct-users-2.8.1"><span class="toc-item-num">2.8.1&nbsp;&nbsp;</span>Perform the bin assignment and estimate distinct users</a></span></li><li><span><a href="#Preview-user-count-map" data-toc-modified-id="Preview-user-count-map-2.8.2"><span class="toc-item-num">2.8.2&nbsp;&nbsp;</span>Preview user count map</a></span></li></ul></li><li><span><a href="#C:-Estimated-User-Days" data-toc-modified-id="C:-Estimated-User-Days-2.9"><span class="toc-item-num">2.9&nbsp;&nbsp;</span>C: Estimated User Days</a></span></li></ul></li><li><span><a href="#Prepare-methods" data-toc-modified-id="Prepare-methods-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Prepare methods</a></span></li><li><span><a href="#Plotting-worldmaps:-Post-Count,-User-Count-and-User-Days" data-toc-modified-id="Plotting-worldmaps:-Post-Count,-User-Count-and-User-Days-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Plotting worldmaps: Post Count, User Count and User Days</a></span></li><li><span><a href="#Save-&amp;-load-intermediate-and-benchmark-data" data-toc-modified-id="Save-&amp;-load-intermediate-and-benchmark-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Save &amp; load intermediate and benchmark data</a></span><ul class="toc-item"><li><span><a href="#Load-&amp;-store-results-from-and-to-CSV" data-toc-modified-id="Load-&amp;-store-results-from-and-to-CSV-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Load &amp; store results from and to CSV</a></span></li><li><span><a href="#Load-&amp;-plot-pickled-dataframe" data-toc-modified-id="Load-&amp;-plot-pickled-dataframe-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Load &amp; plot pickled dataframe</a></span></li></ul></li><li><span><a href="#Close-DB-connection-&amp;-Create-notebook-HTML" data-toc-modified-id="Close-DB-connection-&amp;-Create-notebook-HTML-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Close DB connection &amp; Create notebook HTML</a></span></li><li><span><a href="#Interpretation-of-results" data-toc-modified-id="Interpretation-of-results-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Interpretation of results</a></span></li></ul></div>
<!-- #endregion -->

# Introduction

Based on data from YFCC100m dataset, this Notebook explores a privacy-aware processing example for visualizing frequentation patterns in a 100x100km Grid (worldwide).

This is the third notebook in a series of four notebooks:

* 1) the [Preparations (01_preparations.ipynb)](01_preparations.html) Basic preparations for processing YFCC100m, explains basic concepts and tools for working with the lbsn data
* 2) the [RAW Notebook (02_yfcc_gridagg_raw.ipynb)](02_yfcc_gridagg_raw.html) demonstrates how a typical grid-based visualization looks like when using the **raw** lbsn structure and  
* 3) the [HLL Notebook (03_yfcc_gridagg_hll.ipynb)](03_yfcc_gridagg_hll.html) demonstrates the same visualization using the privacy-aware **hll** lbsn structure  
* 4) the [Interpretation (04_interpretation_interactive_maps.ipynb)](04_interpretation.html) illustrates how to create interactive graphics for comparison of raw and hll results

This notebook includes code parts and examples that have nothing to do with HyperLogLog. Our goal was to illustrate a complete typical visualization pipeline, from reading data to processing to visualization. There're additional steps included such as archiving intermediate results or creating an alternative interactive visualization. At the various parts, we discuss advantages and disadvantages of the privacy-aware data structure compared to working with raw data.

In this Notebook, we describe a complete visualization pipeline, exploring worldwide frequentation patterns from YFCC dataset based on a 100x100km grid. In addition to the steps listed in the raw notebook, this notebooks describes, among other aspects:

* get data from LBSN hll db (PostgreSQL select)  
* store hll data to CSV, load from CSV   
* incremental union of hll sets
* estimated cardinality for metrics postcount, usercount and userdays 
* measure timing of different steps, to compare processing time with raw-dataset approach  
* load and store intermediate results from and to \*.pickle and \*.CSV
* create benchmark data to be published

**System requirements**

The Notebook is configured to run on a computer with **8 GB of Memory** (minimum).

If more is available, you may increase the `chunk_size` parameter (Default is 5000000 records per chunk) to improve speed.

**Additional Notes**:

Use **Shift+Enter** to walk through the Notebook


# Preparations


## Load dependencies

Dependencies are already defined in `02_yfcc_gridagg_raw.ipynb`. We're loading the jupytext python converted version of this notebook to the main namespace, which will make all methods and parameters available here.

```python
import sys
from pathlib import Path
module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)

from _02_yfcc_gridagg_raw import *
from modules import tools, preparations
```

## Parameters

Parameters from the first notebook are available through import. They can be overridden here.

<!-- #raw tags=["active-ipynb"] -->
GRID_SIZE_METERS = 100000 # the size of grid cells in meters 
                          # (spatial accuracy of worldwide measurement)
CHUNK_SIZE = 5000000      # process x number of hll records per chunk.
                          # Increasing this number will consume more memory,
                          # but reduce processing time because less SQL queries
                          # are needed.
<!-- #endraw -->

Activate autoreload of changed python files:

```python tags=["active-ipynb"]
%load_ext autoreload
%autoreload 2
```

Load memory profiler extension

```python tags=["active-ipynb"]
%load_ext memory_profiler
```

Plot used package versions for future use:

```python tags=["hide_code", "active-ipynb"]
root_packages = [
        'geoviews', 'geopandas', 'pandas', 'numpy', 'cloudpickle',
        'matplotlib', 'shapely', 'cartopy', 'holoviews',
        'mapclassify', 'fiona', 'bokeh', 'pyproj', 'ipython',
        'jupyterlab', 'xarray']
preparations.package_report(root_packages)
```

<!-- #region tags=["highlight"] -->
### Connect to database

The password is automatically loaded from `.env` file specified in container setup [hlldb](https://gitlab.vgiscience.de/lbsn/databases/hlldb).

The docker stack contains a full backup of the YFCC database converted to the privacy-aware datastructure. In this Notebook, we're only working with a small part of the data from the table `spatial.latlng`.
<!-- #endregion -->

Define credentials as environment variables

```python tags=["active-ipynb"]
db_user = "postgres"
db_pass = os.getenv('POSTGRES_PASSWORD')
# set connection variables
db_host = "hlldb"
db_port = "5432"
db_name = "hlldb"
```

Connect to empty Postgres database running HLL Extension. Note that only `read` privileges are needed.

```python tags=["active-ipynb"]
db_connection = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass
)
db_connection.set_session(readonly=True)
```

Test connection:

```python tags=["active-ipynb"]
db_query = """
    SELECT 1;
"""
# create pandas DataFrame from database data
df = pd.read_sql_query(db_query, db_connection)
display(df.head())
```

For simplicity, the db connection parameters and query are stored in a class:

```python tags=["active-ipynb"]
db_conn = tools.DbConn(db_connection)
db_conn.query("SELECT 1;")
```

## Get data from db and write to CSV

To compare processing speed with the raw notebook, we're also going to save hll data to CSV first. The following records are available from table spatial.latlng:

* distinct latitude and longitude coordinates (clear text), this is the "base" we're working on
* post_hll - approximate post guids stored as hll set
* user_hll - approximate user guids stored as hll set
* date_hll - approximate user days stored as hll set

```python
def get_yfccposts_fromdb(
        chunk_size: int = CHUNK_SIZE) -> List[pd.DataFrame]:
    """Returns spatial.latlng data from db, excluding Null Island"""
    sql = f"""
    SELECT  latitude,
            longitude,
            post_hll,
            user_hll,
            date_hll
    FROM spatial.latlng t1
    WHERE
    NOT ((latitude = 0) AND (longitude = 0));
    """
    # execute query, enable chunked return
    return pd.read_sql(sql, con=db_connection, chunksize=chunk_size)
```

**Execute Query:**

```python tags=["active-ipynb"]
%%time
filename = "yfcc_latlng.csv"
usecols = ["latitude", "longitude", "post_hll", "user_hll", "date_hll"]
if Path(OUTPUT / "csv" / filename).exists():
        print(f"CSV already exists, skipping load from db.. (to reload, delete file)")
else:
    write_chunkeddf_tocsv(
        chunked_df=get_yfccposts_fromdb(),
        filename=filename,
        usecols=usecols)
```

**HLL file size:**

```python tags=["active-ipynb"]
hll_size_mb = Path(OUTPUT / "csv" / "yfcc_latlng.csv").stat().st_size / (1024*1024)
print(f"Size: {hll_size_mb:.2f} MB")
```

## Create Grid

```python tags=["active-ipynb"]
grid, rows, cols = create_grid_df(
    report=True, return_rows_cols=True)
```

```python tags=["active-ipynb"]
grid = grid_to_gdf(grid)
```

Add columns for aggregation

```python tags=["active-ipynb"]
METRICS = ["postcount_est", "usercount_est", "userdays_est"]
```

```python tags=["active-ipynb"]
reset_metrics(grid, METRICS)
display(grid)
```

**Read World geometries data**

```python tags=["active-ipynb"]
%%time
world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'), crs=CRS_WGS)
world = world.to_crs(CRS_PROJ)
```

## Prepare functions

Now that it has been visually verified that the algorithm works. We'll use some of the functions defined in the previous notebook.


## Test with LBSN data


We're going to test the binning of coordinates on a part of the YFCC geotagged images.


Create 2 bins for each axis of existing Mollweide rows/cols grid:

```python tags=["active-ipynb"]
ybins = np.array(rows)
xbins = np.array(cols)
```

Prepare lat/lng tuple of lower left corner and upper right corner to crop sample map:

```python tags=["active-ipynb"]
# Part of Italy and Sicily
bbox_italy = (
    7.8662109375, 36.24427318493909,
    19.31396484375, 43.29320031385282)
bbox = bbox_italy
```

Calculate bounding box with 1000 km buffer. For that, project the bounding Box to Mollweide, apply the buffer, and project back to WGS1984:

```python tags=["active-ipynb"]
# convert to Mollweide
minx, miny = PROJ_TRANSFORMER.transform(
    bbox_italy[0], bbox_italy[1])
maxx, maxy = PROJ_TRANSFORMER.transform(
    bbox_italy[2], bbox_italy[3])
buf = 1000000
# apply buffer and convetr back to WGS1984
min_buf = PROJ_TRANSFORMER_BACK.transform(minx-buf, miny-buf)
max_buf = PROJ_TRANSFORMER_BACK.transform(maxx+buf, maxy+buf)
bbox_italy_buf = [min_buf[0], min_buf[1], max_buf[0], max_buf[1]]
```

Select columns and types for improving speed

```python tags=["highlight", "active-ipynb"]
usecols = ['latitude', 'longitude', 'post_hll']
dtypes = {'latitude': float, 'longitude': float}
reset_metrics(grid, METRICS)
```

### Load data

```python tags=["active-ipynb"]
%%time
df = pd.read_csv(
    OUTPUT / "csv" / "yfcc_latlng.csv", usecols=usecols, dtype=dtypes, encoding='utf-8')
print(len(df))
```

Execute and count number of posts in the bounding box:

```python tags=["active-ipynb"]
%%time
filter_df_bbox(df=df, bbox=bbox_italy_buf)
print(f"There're {len(df):,.0f} YFCC distinct lat-lng coordinates located within the bounding box.")
df.head()
```

### Project coordinates to Mollweide

```python tags=["active-ipynb"]
%%time
proj_df(df)
print(f'Projected {len(df.values)} coordinates')
df.head()
```

### Perform the bin assignment

```python tags=["active-ipynb"]
%%time
xbins_match, ybins_match = get_best_bins(
    search_values_x=df['x'].to_numpy(),
    search_values_y=df['y'].to_numpy(),
    xbins=xbins, ybins=ybins)
```

```python tags=["active-ipynb"]
len(xbins_match)
```

```python tags=["active-ipynb"]
xbins_match[:10]
```

```python tags=["active-ipynb"]
ybins_match[:10]
```

## A: Estimated Post Count per grid

Attach target bins to original dataframe. The `:` means: modify all values in-place

```python tags=["active-ipynb"]
df.loc[:, 'xbins_match'] = xbins_match
df.loc[:, 'ybins_match'] = ybins_match
# set new index column
df.set_index(['xbins_match', 'ybins_match'], inplace=True)
# drop x and y columns not needed anymore
df.drop(columns=['x', 'y'], inplace=True)
```

```python tags=["active-ipynb"]
df.head()
```

The next step is to union hll sets and (optionally) return the cardinality (the number of distinct elements). This can be done by connecting to a postgres database with HLL extension installed. There's a [python package available](https://github.com/AdRoll/python-hll) for HLL calculations, but it is in a very early stage of development. For simplicity, we're using our `hlldb` here, but it is equally possible to connect to an empty Postgres DB running Citus HLL such as [pg-hll-empty docker container](https://gitlab.vgiscience.de/lbsn/databases/pg-hll-empty).

```python tags=["highlight_red"]
def union_hll(
    hll_series: pd.Series, cardinality: bool = True) -> pd.Series:
    """HLL Union and (optional) cardinality estimation from series of hll sets
    based on group by composite index.

        Args:
        hll_series: Indexed series (bins) of hll sets. 
        cardinality: If True, returns cardinality (counts). Otherwise,
            the unioned hll set will be returned.
            
    The method will combine all groups of hll sets first,
        in a single SQL command. Union of hll hll-sets belonging 
        to the same group (bin) and (optionally) returning the cardinality 
        (the estimated count) per group will be done in postgres.
    
    By utilizing Postgres´ GROUP BY (instead of, e.g. doing 
        the group with numpy), it is possible to reduce the number
        of SQL calls to a single run, which saves overhead 
        (establishing the db connection, initializing the SQL query 
        etc.). Also note that ascending integers are used for groups,
        instead of their full original bin-ids, which also reduces
        transfer time.
    
    cardinality = True should be used when calculating counts in
        a single pass.
        
    cardinality = False should be used when incrementally union
        of hll sets is required, e.g. due to size of input data.
        In the last run, set to cardinality = True.
    """
    # group all hll-sets per index (bin-id)
    series_grouped = hll_series.groupby(
        hll_series.index).apply(list)
    # From grouped hll-sets,
    # construct a single SQL Value list;
    # if the following nested list comprehension
    # doesn't make sense to you, have a look at
    # spapas.github.io/2016/04/27/python-nested-list-comprehensions/
    # with a decription on how to 'unnest'
    # nested list comprehensions to regular for-loops
    hll_values_list = ",".join(
        [f"({ix}::int,'{hll_item}'::hll)" 
         for ix, hll_items
         in enumerate(series_grouped.values.tolist())
         for hll_item in hll_items])
    # Compilation of SQL query,
    # depending on whether to return the cardinality
    # of unioned hll or the unioned hll
    return_col = "hll_union"
    hll_calc_pre = ""
    hll_calc_tail = "AS hll_union"
    if cardinality:
        # add sql syntax for cardinality 
        # estimation
        # (get count distinct from hll)
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
    # to merge values back to grouped dataframe,
    # first reset index to ascending integers
    # matching those of the returned df;
    # this will turn series_grouped into a DataFrame;
    # the previous index will still exist in column 'index'
    df_grouped = series_grouped.reset_index()
    # drop hll sets not needed anymore
    df_grouped.drop(columns=[hll_series.name], inplace=True)
    # append hll_cardinality counts 
    # using matching ascending integer indexes
    df_grouped.loc[df.index, return_col] = df[return_col]
    # set index back to original bin-ids
    df_grouped.set_index("index", inplace=True)
    # split tuple index to produce
    # the multiindex of the original dataframe
    # with xbin and ybin column names
    df_grouped.index = pd.MultiIndex.from_tuples(
        df_grouped.index, names=['xbin', 'ybin'])
    # return column as indexed pd.Series
    return df_grouped[return_col]
```

Optionally, split dataframe into chunks, so we're not the exceeding memory limit (e.g. use if memory < 16GB). A chunk size of 1 Million records is suitable for a computer with about 8 GB of memory and optional sparse HLL mode enabled. If sparse mode is disabled, decrease chunk_size accordingly, to compensate for increased space.

```python tags=["active-ipynb"]
%%time
chunked_df = [
    df[i:i+CHUNK_SIZE] for i in range(0, df.shape[0], CHUNK_SIZE)]
```

```python tags=["active-ipynb"]
chunked_df[0].head()
```

To test, process the first chunk:

```python tags=["highlight_red", "active-ipynb"]
%%time
cardinality_series = union_hll(chunked_df[0]["post_hll"])
```

```python tags=["active-ipynb"]
cardinality_series.head()
```

Remove possibly existing result column in grid from previous run:

```python tags=["active-ipynb"]
reset_metrics(grid, ["postcount_est"], setzero=True)
```

Append Series with calculated counts to grid (as new column) based on index match:

```python tags=["active-ipynb"]
grid.loc[cardinality_series.index, 'postcount_est'] = cardinality_series
```

```python tags=["active-ipynb"]
grid[grid["postcount_est"] > 0].head()
```

**Process all chunks:**

The caveat here is to incrementally union hll sets until all records have been processed. On the last loop, instruct the hll worker to return the cardinality instead of the unioned hll set.


First, define method to join cardinality to grid

```python tags=["highlight"]
# reference metric names and column names
COLUMN_METRIC_REF = {
        "postcount_est":"post_hll",
        "usercount_est":"user_hll",
        "userdays_est":"date_hll"}

def join_df_grid(
    df: pd.DataFrame, grid: gp.GeoDataFrame,
    metric: str = "postcount_est",
    cardinality: bool = True,
    column_metric_ref: Dict[str,str] = COLUMN_METRIC_REF):
    """Union HLL Sets and estimate postcount per 
    grid bin from lat/lng coordinates
    
        Args:
        df: A pandas dataframe with latitude and 
            longitude columns in WGS1984
        grid: A geopandas geodataframe with indexes 
            x and y (projected coordinates) and grid polys
        metric: target column for estimate aggregate.
            Default: postcount_est.
        cardinality: will compute cardinality of unioned
            hll sets. Otherwise, unioned hll sets will be 
            returned for incremental updates.
    """
    # optionally, bin assigment of projected coordinates,
    # make sure to not bin twice:
    # x/y columns are removed after binning
    if 'x' in df.columns:
        bin_coordinates(df, xbins, ybins)
        # set index column
        df.set_index(
            ['xbins_match', 'ybins_match'], inplace=True)
    # union hll sets and 
    # optional estimate count distincts (cardinality)
    column = column_metric_ref.get(metric)
    # get series with grouped hll sets
    hll_series = df[column]
    # union of hll sets:
    # to allow incremental union of already merged data
    # and new data, concatenate series from grid and new df
    # only if column with previous hll sets already exists
    if metric in grid.columns:
        # remove nan values from grid and
        # rename series to match names
        hll_series = pd.concat(
            [hll_series, grid[metric].dropna()]
            ).rename(column)
    cardinality_series = union_hll(
        hll_series, cardinality=cardinality)
    # add unioned hll sets/computed cardinality to grid
    grid.loc[
        cardinality_series.index, metric] = cardinality_series
    if cardinality:
        # set all remaining grid cells
        # with no data to zero and
        # downcast column type from float to int
        grid[metric] = grid[metric].fillna(0).astype(int)
```

Define method to process chunks:

```python tags=["highlight"]
def join_chunkeddf_grid(
    chunked_df: List[pd.DataFrame], grid: gp.GeoDataFrame,
    metric: str = "postcount_est", chunk_size: int = CHUNK_SIZE,
    benchmark_data: Optional[bool] = None,
    column_metric_ref: Dict[str,str] = COLUMN_METRIC_REF):
    """Incremental union of HLL Sets and estimate postcount per 
    grid bin from chunked list of dataframe records. Results will
    be stored in grid.
    
    Args:
    chunked_df: A list of (chunked) dataframes with latitude and 
        longitude columns in WGS1984
    grid: A geopandas geodataframe with indexes 
        x and y (projected coordinates) and grid polys
    metric: target column for estimate aggregate.
        Default: postcount_est.
    benchmark_data: If True, will not remove HLL sketches after
        final cardinality estimation.
    column_metric_ref: Dictionary containing references of 
        metrics to df columns.
    """
    reset_metrics(grid, [metric])
    for ix, chunk_df in enumerate(chunked_df):
        # compute cardinality only on last iteration
        cardinality = False
        if ix == len(chunked_df)-1:
            cardinality = True
        column = column_metric_ref.get(metric)
        # get series with grouped hll sets
        hll_series = chunk_df[column]
        if metric in grid.columns:
            # merge existing hll sets with new ones
            # into one series (with duplicate indexes);
            # remove nan values from grid and
            # rename series to match names
            hll_series = pd.concat(
                [hll_series, grid[metric].dropna()]
                ).rename(column)
        cardinality_series = union_hll(
            hll_series, cardinality=cardinality)
        if benchmark_data and (ix == len(chunked_df)-1):
            # only if final hll sketches need to
            # be kept for benchmarking:
            # do another union, without cardinality
            # estimation, and store results
            # in column "metric"_hll
            hll_sketch_series = union_hll(
                hll_series, cardinality=False)
            grid.loc[
                hll_sketch_series.index,
                f'{metric.replace("_est","_hll")}'] = hll_sketch_series
        # add unioned hll sets/computed cardinality to grid
        grid.loc[
            cardinality_series.index, metric] = cardinality_series
        if cardinality:
            # set all remaining grid cells
            # with no data to zero and
            # downcast column type from float to int
            grid[metric] = grid[metric].fillna(0).astype(int)
        clear_output(wait=True)
        print(f'Mapped ~{(ix+1)*chunk_size} coordinates to bins')
```

```python tags=["active-ipynb"]
join_chunkeddf_grid(chunked_df, grid, chunk_size=CHUNK_SIZE)
```

All distinct counts are now attached to the bins of the grid:

```python tags=["active-ipynb"]
grid[grid["postcount_est"]>10].head()
```

### Preview post count map


Use headtail_breaks classification scheme because it is specifically suited to map long tailed data, see [Jiang 2013](https://arxiv.org/pdf/1209.2801)

* Jiang, B. (August 01, 2013). Head/Tail Breaks: A New Classification Scheme for Data with a Heavy-Tailed Distribution. The Professional Geographer, 65, 3, 482-494.

```python tags=["active-ipynb"]
# global legend font size setting
plt.rc('legend', **{'fontsize': 16})
```

```python tags=["active-ipynb"]
save_plot(
    grid=grid, title='Estimated Post Count',
    column='postcount_est', save_fig='postcount_sample_est.png',
    bbox=bbox_italy, world=world)
```

## B: Estimated User Count per grid

When using HLL, aggregation of user_guids or user_days takes the same amount of time (unlike when working with original data, where memory consumption increases significantly). We'll only need to update the columns that are loaded from the database:

```python tags=["active-ipynb"]
usecols = ['latitude', 'longitude', 'user_hll']
```

Adjust method for stream-reading from CSV in chunks:

```python tags=["highlight", "active-ipynb"]
iter_csv = pd.read_csv(
    OUTPUT / "csv" / "yfcc_latlng.csv", usecols=usecols, iterator=True,
    dtype=dtypes, encoding='utf-8', chunksize=CHUNK_SIZE)
```

```python tags=["active-ipynb"]
%%time
# filter
chunked_df = [
    filter_df_bbox( 
        df=chunk_df, bbox=bbox_italy_buf, inplace=False)
    for chunk_df in iter_csv]

# project
projected_cnt = 0
for chunk_df in chunked_df:
    projected_cnt += len(chunk_df)
    proj_report(
        chunk_df, projected_cnt, inplace=True)

chunked_df[0].head()
```

### Perform the bin assignment and estimate distinct users

```python tags=["active-ipynb"]
%%time
bin_chunked_coordinates(chunked_df)
chunked_df[0].head()
```

Union HLL Sets per grid-id and calculate cardinality (estimated distinct user count):

```python tags=["active-ipynb"]
join_chunkeddf_grid(
    chunked_df=chunked_df, grid=grid, metric="usercount_est")
```

```python tags=["active-ipynb"]
grid[grid["usercount_est"]> 0].head()
```

Look at this. There're many polygons were thounsands of photos have been created by only few users. Lets see how this affects our test map..


### Preview user count map

```python tags=["active-ipynb"]
save_plot(
    grid=grid, title='Estimated User Count',
    column='usercount_est', save_fig='usercount_sample_est.png',
    bbox=bbox_italy, world=world)
```

## C: Estimated User Days

Usually, due to the [Count Distinct Problem](https://en.wikipedia.org/wiki/Count-distinct_problem) increasing computation times will apply for more complex distinct queries. This is not the case when using HLL. Any count distinct (postcount, usercount etc.) requires the same amount of time. A useful metric introduced by Wood et al. (2013) is User Days, which lies inbetween Post Count and User Count because Users may be counted more than once if they visited the location on consecutive days. User Days particularly allows capturing the difference between local and tourist behaviour patterns. The rationale here is that locals visit few places very often. In contrast, tourists visit many places only once.

The sequence of commands for userdays is exactly the same as for postcount and usercount above.

```python tags=["active-ipynb"]
usecols = ['latitude', 'longitude', 'date_hll']
```

```python
def read_project_chunked(filename: str,
    usecols: List[str], chunk_size: int = CHUNK_SIZE,
    bbox: Tuple[float, float, float, float] = None,
    output: Path = OUTPUT,
    dtypes = None) -> List[pd.DataFrame]:
    """Read data from csv, optionally clip to bbox and projet"""
    if dtypes is None:
        dtypes = {'latitude': float, 'longitude': float}
    iter_csv = pd.read_csv(
        output / "csv" / filename, usecols=usecols, iterator=True,
        dtype=dtypes, encoding='utf-8', chunksize=chunk_size)
    if bbox:
        chunked_df = [filter_df_bbox( 
            df=chunk_df, bbox=bbox, inplace=False)
        for chunk_df in iter_csv]
    else:
        chunked_df = [chunk_df for chunk_df in iter_csv]
    # project
    projected_cnt = 0
    for chunk_df in chunked_df:
        projected_cnt += len(chunk_df)
        proj_report(
            chunk_df, projected_cnt, inplace=True)
    return chunked_df
```

Run:

```python tags=["active-ipynb"]
%%time
chunked_df = read_project_chunked(
    filename="yfcc_latlng.csv",
    usecols=usecols,
    bbox=bbox_italy_buf)
chunked_df[0].head()
```

```python tags=["active-ipynb"]
%%time
bin_chunked_coordinates(chunked_df)
```

```python tags=["active-ipynb"]
join_chunkeddf_grid(
    chunked_df=chunked_df, grid=grid, metric="userdays_est")
```

```python tags=["active-ipynb"]
chunked_df[0].head()
```

```python tags=["active-ipynb"]
grid[grid["userdays_est"]> 0].head()
```

```python tags=["active-ipynb"]
save_plot(
    grid=grid, title='Estimated User Days',
    column='userdays_est', save_fig='userdays_sample_est.png',
    bbox=bbox_italy, world=world)
```

There're other approaches for further reducing noise. For example, to reduce the impact of automatic capturing devices (such as webcams uploading x pictures per day), a possibility is to count distinct **userlocations**. For userlocations metric, a user would be counted multiple times per grid bin only for pictures with different lat/lng. Or the number of distinct **userlocationdays** (etc.). These metrics easy to implement using hll, but would be quite difficult to compute using raw data.


# Prepare methods

Lets summarize the above code in a few methods:


**Plotting preparation**

The method below utilizes many of the methods defined for raw data processing.

```python tags=["highlight"]
def load_plot(
    grid: gp.GeoDataFrame, title: str, inverse: bool = None,
    metric: str = "postcount_est", store_fig: str = None, store_pickle: str = None,
    chunk_size: int = CHUNK_SIZE, benchmark_data: Optional[bool] = None,
    column_metric_ref: Dict[str,str] = COLUMN_METRIC_REF):
    """Load data, bin coordinates, estimate distinct counts (cardinality) and plot map
    
        Args:
        data: Path to read input CSV
        grid: A geopandas geodataframe with indexes x and y 
            (projected coordinates) and grid polys
        title: Title of the plot
        inverse: If True, inverse colors (black instead of white map)
        metric: target column for aggregate. Default: postcount_est.
        store_fig: Provide a name to store figure as PNG. Will append 
            '_inverse.png' if inverse=True.
        store_pickle: Provide a name to store pickled dataframe
            with aggregate counts to disk
        chunk_size: chunk processing into x records per chunk
        benchmark_data: If True, hll_sketches will not be removed 
            after final estimation of cardinality
    """
    usecols = ['latitude', 'longitude']
    column = column_metric_ref.get(metric)
    usecols.append(column)
    # get data from csv
    chunked_df = read_project_chunked(
        filename="yfcc_latlng.csv",
        usecols=usecols)
    # bin coordinates
    bin_chunked_coordinates(chunked_df)
    # reset metric column
    reset_metrics(grid, [metric], setzero=False)
    print("Getting cardinality per bin..")
    # union hll sets per chunk and 
    # calculate distinct counts on last iteration
    join_chunkeddf_grid(
        chunked_df=chunked_df, grid=grid,
        metric=metric, chunk_size=chunk_size,
        benchmark_data=benchmark_data)
    # store intermediate data
    if store_pickle:
        print("Storing aggregate data as pickle..")
        grid.to_pickle(output / "pickles" / store_pickle)
    print("Plotting figure..")
    plot_figure(grid=grid, title=title, inverse=inverse, metric=metric, store_fig=store_fig)
```

# Plotting worldmaps: Post Count, User Count and User Days

Plot worldmap for each datasource

```python tags=["active-ipynb"]
reset_metrics(grid, ["postcount_est", "usercount_est", "userdays_est"])
```

```python tags=["active-ipynb"]
%%time
%%memit
load_plot(
    grid, title=f'Estimated YFCC Post Count per {int(GRID_SIZE_METERS/1000)}km grid',
    inverse=False, store_fig="yfcc_postcount_est.png", benchmark_data=True)
```

```python tags=["active-ipynb"]
%%time
%%memit
load_plot(
    grid, title=f'Estimated YFCC User Count per {int(GRID_SIZE_METERS/1000)}km grid',
    inverse=False, store_fig="yfcc_usercount_est.png",
    metric="usercount_est", benchmark_data=True)
```

```python tags=["active-ipynb"]
%%time
%%memit
load_plot(
    grid, title=f'Estimated YFCC User Days per {int(GRID_SIZE_METERS/1000)}km grid',
    inverse=False, store_fig="yfcc_userdays_est.png",
    metric="userdays_est", benchmark_data=True)
```

Have a look at the final `grid` with the estimated cardinality for postcount, usercount and userdays

it is possible to make an immediate validation of the numbers by verifying that `postcount` >= `userdays` >= `usercount`. On very rare occasions and edge cases, this may invalidate due to the estimation error of 3 to 5% of HyperLogLog derived cardinality.

```python tags=["active-ipynb"]
grid[grid["postcount_est"]>1].drop(
    ['geometry', 'usercount_hll', 'postcount_hll', 'userdays_hll'], axis=1, errors="ignore").head()
```

Final HLL Sets are also available, as benchmark data, in columns `usercount_hll`, `postcount_hll`, `userdays_hll` columns:

```python tags=["active-ipynb"]
grid[grid["postcount_est"]>1].drop(
    ['geometry', 'usercount_est', 'postcount_est', 'userdays_est'], axis=1, errors="ignore").head()
```

# Save & load intermediate and benchmark data
## Load & store results from and to CSV

To export only aggregate counts (postcount, usercount) to CSV (e.g. for archive purposes):


**Store results to CSV for archive purposes:**


Convert/store to CSV (aggregate columns and indexes only):

```python tags=["active-ipynb"]
grid_agg_tocsv(grid, "yfcc_all_est.csv", metrics=["postcount_est", "usercount_est", "userdays_est"])
```

**Store results as benchmark data (with hll sketches):**

**a)** Published data scenario

As a minimal protection against intersection attacks on published data, only export
hll sets with cardinality > 100.

```python tags=["active-ipynb"]
grid_agg_tocsv(
    grid[grid["usercount_est"]>100], "yfcc_all_est_benchmark.csv", 
    metrics = ["postcount_est", "usercount_est", "userdays_est",
               "usercount_hll", "postcount_hll", "userdays_hll"])
```

Size of benchmark data:

```python tags=["active-ipynb"]
benchmark_size_mb = Path(OUTPUT / "csv" / "yfcc_all_est_benchmark.csv").stat().st_size / (1024*1024)
print(f"Size: {benchmark_size_mb:.2f} MB")
```

**b)** Internal database scenario

Export all hll sets, as a means to demonstrate internal data compromisation.

```python tags=["active-ipynb"]
grid_agg_tocsv(
    grid, "yfcc_all_est_benchmark_internal.csv", 
    metrics = ["postcount_est", "usercount_est", "userdays_est",
               "usercount_hll", "postcount_hll", "userdays_hll"])
```

Size of benchmark data:

```python tags=["active-ipynb"]
benchmark_size_mb = Path(OUTPUT / "csv" / "yfcc_all_est_benchmark_internal.csv").stat().st_size / (1024*1024)
print(f"Size: {benchmark_size_mb:.2f} MB")
```

**Load data from CSV:**


To create a new grid and load aggregate counts from CSV:

```python tags=["active-ipynb"]
grid = grid_agg_fromcsv(
    OUTPUT / "csv" / "yfcc_all_est.csv",
    columns=["xbin", "ybin", "postcount_est", "usercount_est", "userdays_est"])
```

## Load & plot pickled dataframe


Loading (geodataframe) using [pickle](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html#pandas.read_pickle). This is the easiest way to store intermediate data, but may be incompatible [if package versions change](https://stackoverflow.com/questions/6687262/how-do-i-know-which-versions-of-pickle-a-particular-version-of-python-supports). If loading pickles does not work, a workaround is to load data from CSV and re-create pickle data, which will be compatible with used versions.


**Store results using pickle for later resuse:**

```python tags=["active-ipynb"]
grid.to_pickle(OUTPUT / "pickles" / "yfcc_all_est.pkl")
```

**Load pickled dataframe:**

```python tags=["active-ipynb"]
%%time
grid = pd.read_pickle(OUTPUT / "pickles" / "yfcc_all_est.pkl")
```

Then use plot_figure on dataframe to plot with new parameters, e.g. plot inverse:

```python tags=["active-ipynb"]
plot_figure(grid, "Pickle Test", inverse=True, metric="postcount_est")
```

**To merge results of raw and hll dataset:**

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

# Close DB connection & Create notebook HTML

```python tags=["active-ipynb"]
db_connection.close ()
```

```python tags=["active-ipynb"]
!jupyter nbconvert --to html_toc \
    --output-dir=../out/html ./03_yfcc_gridagg_hll.ipynb \
    --template=../nbconvert.tpl \
    --ExtractOutputPreprocessor.enabled=False # create single output file
```

# Interpretation of results

The last part of the tutorial will look at ways to improve interpretation of results. Interactive bokeh maps and widget tab display are used to make comparison of raw and hll results easier. Follow in in [04_interpretation.ipynb](04_interpretation.ipynb)

