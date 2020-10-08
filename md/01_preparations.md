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

# Privacy-aware grid aggregation: Preparations (YFCC100m data)<a class="tocSkip">

<!-- #region toc=true -->
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#The-YFCC100m-dataset" data-toc-modified-id="The-YFCC100m-dataset-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>The YFCC100m dataset</a></span></li><li><span><a href="#Structure" data-toc-modified-id="Structure-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Structure</a></span></li><li><span><a href="#Importing-YFCC100m" data-toc-modified-id="Importing-YFCC100m-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Importing YFCC100m</a></span><ul class="toc-item"><li><span><a href="#Download-of-the-YFCC100m-data" data-toc-modified-id="Download-of-the-YFCC100m-data-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Download of the YFCC100m data</a></span></li><li><span><a href="#Importing-the-YFCC100m-to-Postgres" data-toc-modified-id="Importing-the-YFCC100m-to-Postgres-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Importing the YFCC100m to Postgres</a></span></li></ul></li><li><span><a href="#Prepare-RAW-data-for-grid-aggregation" data-toc-modified-id="Prepare-RAW-data-for-grid-aggregation-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Prepare RAW data for grid aggregation</a></span><ul class="toc-item"><li><span><a href="#Defining-the-Query" data-toc-modified-id="Defining-the-Query-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Defining the Query</a></span></li><li><span><a href="#Connect-to-rawdb" data-toc-modified-id="Connect-to-rawdb-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Connect to rawdb</a></span><ul class="toc-item"><li><span><a href="#Load-dependencies" data-toc-modified-id="Load-dependencies-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Load dependencies</a></span></li><li><span><a href="#Establish-connection" data-toc-modified-id="Establish-connection-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>Establish connection</a></span></li><li><span><a href="#Create-Query-Schema" data-toc-modified-id="Create-Query-Schema-5.2.3"><span class="toc-item-num">5.2.3&nbsp;&nbsp;</span>Create Query Schema</a></span></li><li><span><a href="#Prepare-query-and-cryptographic-hashing" data-toc-modified-id="Prepare-query-and-cryptographic-hashing-5.2.4"><span class="toc-item-num">5.2.4&nbsp;&nbsp;</span>Prepare query and cryptographic hashing</a></span></li></ul></li><li><span><a href="#Apply-RAW-query" data-toc-modified-id="Apply-RAW-query-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Apply RAW query</a></span></li></ul></li><li><span><a href="#Convert-data-from-rawdb-to-hlldb" data-toc-modified-id="Convert-data-from-rawdb-to-hlldb-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Convert data from rawdb to hlldb</a></span><ul class="toc-item"><li><span><a href="#Prepare-rawdb" data-toc-modified-id="Prepare-rawdb-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Prepare rawdb</a></span></li><li><span><a href="#Connect-hlldb-to-rawdb" data-toc-modified-id="Connect-hlldb-to-rawdb-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Connect hlldb to rawdb</a></span></li><li><span><a href="#Prepare-conversion-of-raw-data-to-hll" data-toc-modified-id="Prepare-conversion-of-raw-data-to-hll-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Prepare conversion of raw data to hll</a></span></li><li><span><a href="#HyperLogLog-parameters" data-toc-modified-id="HyperLogLog-parameters-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>HyperLogLog parameters</a></span></li><li><span><a href="#Aggregation-step:-Convert-data-to-Hll" data-toc-modified-id="Aggregation-step:-Convert-data-to-Hll-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Aggregation step: Convert data to Hll</a></span></li></ul></li><li><span><a href="#Visualize-data" data-toc-modified-id="Visualize-data-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Visualize data</a></span></li></ul></div>
<!-- #endregion -->

# Introduction

Based on data from YFCC100m dataset, this Notebook series explores a privacy-aware processing example for visualizing frequentation patterns in a 100x100km Grid (worldwide).

This is the first notebook in a series of four notebooks:

* 1) the [Preparations (01_preparations.ipynb)](01_preparations.html) Preparations for importing and pre-processing YFCC100m.
* 2) the [RAW Notebook (02_yfcc_gridagg_raw.ipynb)](02_yfcc_gridagg_raw.html) demonstrates how a typical grid-based visualization looks like when using the **raw** lbsn structure and  
* 3) the [HLL Notebook (03_yfcc_gridagg_hll.ipynb)](03_yfcc_gridagg_hll.html) demonstrates the same visualization using the privacy-aware **hll** lbsn structure  
* 4) the [Interpretation (04_interpretation.ipynb)](04_interpretation.html) illustrates how to create interactive graphics for comparison of raw and hll results; intersection of published data.

This notebook includes the following steps:

* getting the YFCC dataset (CSV)
* importing the YFCC dataset to Postgres (raw database), to a common format that makes it easier to work with the data
* preparation of raw data that is used for the grid aggregation
* conversion of raw data to hll format

# Preparations

For easier replication, the notebooks make use of several docker containers. If you want to follow the notebooks without docker, you need to setup services (Postgres with citus HLL, jupyter lab) based on your own system configuration.


## Parameters

Define global settings

```python
# GeoHash precision level
# to pre-aggregate spatial data (coordinates)
GEOHASH_PRECISION = 5
```

```python
from pathlib import Path
# define path to output directory (figures etc.)
OUTPUT = Path.cwd().parents[0] / "out"
```

**Create paths**

```python
def create_paths(
    output: str = OUTPUT, subfolders = ["html", "pickles", "csv", "figures"]):
    """Create subfolder for results to be stored"""
    output.mkdir(exist_ok=True)
    for subfolder in subfolders:
        Path(OUTPUT / subfolder).mkdir(exist_ok=True)
```

```python
create_paths()
```

Load keys from .env

```python
import os
from dotenv import load_dotenv

dotenv_path = Path.cwd().parents[0] / '.env'
load_dotenv(dotenv_path)
CRYPT_KEY = os.getenv("CRYPT_KEY")
USER_KEY = os.getenv("USER_KEY")
```

<!-- #region -->
## Install dependencies

Below is a summary of requirements to get this notebook running:


If you want to run the notebook yourself, either get the [LBSN JupyterLab Docker](https://gitlab.vgiscience.de/lbsn/tools/jupyterlab), or follow these steps to create an environment in [conda](https://docs.conda.io/en/latest/) for running this notebook. Suggested using `miniconda` (in Windows, use WSL).

```bash
conda create -n yfcc_env -c conda-forge
conda activate yfcc_env
conda config --env --set channel_priority strict
conda config --show channel_priority # verify
# visualization dependencies
conda install -c conda-forge geopandas jupyterlab "geoviews-core=1.8.1" descartes mapclassify jupyter_contrib_nbextensions xarray
# only necessary when using data from db
conda install -c conda-forge python-dotenv psycopg2
```

to upgrade later, use:
```bash
conda upgrade -n yfcc_env --all -c conda-forge
```

Pinning geoviews to 1.8.1 should result in packages installed that are compatible with the code herein.
<!-- #endregion -->

# The YFCC100m dataset

The YFCC100m dataset is a typical example of user-generated content that is made [publicly 
available][1]
for anyone to use. It was published by Flickr in 2014 (Thomee et al. 2013).
The core dataset is distributed as a compressed archive that contains only the metadata for 
about 100 Million photos and videos from Flickr published under a 
Creative Commons License. About 48 Million of the photos are geotagged.

Even if user-generated data is explicitly made public, like in this case,
certain risks to privacy exist. Data may be re-purposed in contexts
not originally anticipated by the users publishing the data. IBM, for example, 
[re-purposed the YFCC100m dataset][3] 
to fuel a facial-recognition project, without the consent of the people in the images.

[1]: https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
[2]: https://www.flickr.com/photos/ayman/14446556792
[3]: https://www.inavateonthenet.net/news/article/ibm-used-flickr-photos-without-consent-for-facial-recognition-project
[4]: https://www.flickr.com/services/api/flickr.photos.geo.photosForLocation.html


# Structure

The core dataset consists of two CSV files of about 14 GB which are hosted on Amazon AWS S3 bucket.
This dataset contains a list of photos and videos and related meta data (titles, tags, timestamps etc.).

An overview of available columns in this dataset is provided by Deng & Li (2018). The table
below contain a summary of the CSV columns.

| Column | Metadata Description                | Example                                                      |
|--------|-------------------------------------|--------------------------------------------------------------|
| 0      | row id                              | 0                                                            |
| 1      | Photo/video identifier              | 6185218911                                                   |
| 2      | User NSID                           | 4e2f7a26a1dfbf165a7e30bdabf7e72a                             |
| 3      | User ID                             | 39019111@N00                                                 |
| 4      | User nickname                       | guckxzs                                                      |
| 5      | Date taken                          | 2012-02-16 09:56:37.0                                        |
| 6      | Date uploaded                       | 1331840483                                                   |
| 7      | Capture device                      | Canon+PowerShot+ELPH+310+HS                                  |
| 8      | Title                               | IMG_0520                                                     |
| 9      | Description  ?                      | My vacation                                                  |
| 10     | User tags (comma-separated)         | canon,canon+powershot+hs+310                                 |
| 11     | Machine tags (comma-separated)      | landscape, hills, water                                      |
| 12     | Longitude                           | -81.804885                                                   |
| 13     | Latitude                            | 24.550558                                                    |
| 14     | Accuracy Level (see [Flickr API][4]) | 12                                                           |
| 15     | Photo/video page URL                | http://www.flickr.com/photos/39089491@N00/6985418911/        |
| 16     | Photo/video download URL            | http://farm8.staticflickr.com/7205/6985418911_df7747990d.jpg |
| 17     | License name                        | Attribution-NonCommercial-NoDerivs License                   |
| 18     | License URL                         | http://creativecommons.org/licenses/by-nc-nd/2.0/            |
| 19     | Photo/video server identifier       | 7205                                                         |
| 20     | Photo/video farm identifier         | 8                                                            |
| 21     | Photo/video secret                  | df7747990d                                                   |
| 22     | Photo/video secret original         | 692d7e0a7f                                                   |
| 23     | Extension of the original photo     | jpg                                                          |
| 24     | Marker (0 ¼ photo, 1 ¼ video)       | 0                                                            |

**Table 1:** Summary of Metadata for each CSV column available in the core dataset (`yfcc100m_dataset.csv`).

Next to this core dataset, several expansion packs have been released that provide additional data:

* Autotags: Auto tags added by deep learning (e.g. people, animals, objects, food, events, architecture, and scenery)
* Places: User provided geotags and automatically associated places.
* Exif: Additional Exif data for each photo

As a means to enrich the spatial information, the places expansion set is available (but is not needed to follow the guides herein).

| Column | Metadata Description                | Example                                                         |
|--------|-------------------------------------|-----------------------------------------------------------------|
| 0      | Photo/video identifier              | 6985418911                                                      |
| 1      | Place reference (null to multiple)  | 24703176:Admiralty:Suburb,24703128:Central+and+Western:Territory|

**Table 2:** Summary of Metadata for each CSV column available in the places expansion dataset (`yfcc100m_places.csv`).


# Importing YFCC100m  

To follow the examples used in this guide, follow the steps below.


## Download of the YFCC100m data

Please follow the instructions provided on the [official site][1].

> **Getting the YFCC100M**: The dataset can be requested at 
> [Yahoo Webscope][2]. You will need 
> to create a Yahoo account if you do not have one already, and once logged in you 
> will find it straightforward to submit the request for the YFCC100M. Webscope will 
> ask you to tell them what your plans are with the dataset, which helps them justify 
> the existence of their academic outreach program and allows them to keep offering 
> datasets in the future. Unlike other datasets available at Webscope, the YFCC100M 
> does not require you to be a student or faculty at an accredited university, so you 
> will be automatically approved.

[1]: https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
[2]: https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67


## Importing the YFCC100m to Postgres

<!-- #region -->
Any conversion from one structure to another requires the definition of mapping rules.

To demonstrate mapping of arbitrary LBSN data to the common LBSN structure scheme, 
we have built [lbsntransform][3], a python package
that includes several pre-defined mapping sets.

Note: Have [a look][4] at the exact mapping criteria for the Flickr YFCC100M dataset. The 
package also contains examples for [other mappings][5] (e.g. Twitter, Facebook Places),
which can be extended further.

You'll also need a Postgres Database with the SQL Implementation of the LBSN Structure.

The easiest way is to use [full-stack-lbsn][6], a shell script that starts the following docker
services:

* [rawdb][7]: A ready to use Docker Container with the SQL 
  implementation of LBSN Structure
* [hlldb][8]: A ready to use Docker Container with a privacy-aware 
  version of LBSN Structure, e.g. for visual analytics
* [pgadmin][9]: A web-based PostgreSQL database interface.
* [jupyterlab][10]: A modern web-based user interface for python visual analytics.


Tip: If you're familiar with [git] and [docker], 
you can also clone the above repositories separately
and start individual services as needed.

**Windows user?**
If you're working with Windows, [full-stack-lbsn][6]
will only work in Windows Subsystem for Linux (WSL). Even if it is possible to run
Docker containers natively in Windows, we strongly recommend using [WSL] or [WSL2].
    
After you have started the [rawdb][7] docker container,
import Flickr YFCC CSVs to the database using [lbsntransform][3].

```bash
lbsntransform --origin 21 \
    --file_input \
    --input_path_url "/data/flickr_yfcc100m/" \
    --dbpassword_output "sample-password" \
    --dbuser_output "postgres" \
    --dbserveraddress_output "127.0.0.1:15432" \
    --dbname_output "rawdb" \
    --csv_delimiter $'\t' \
    --file_type "csv" \
    --zip_records
```

* **input_path_url**: The path to the folder where 
  `yfcc100m_places.csv` and `yfcc100m_dataset.csv` are saved.
* **dbpassword_output**: Provide the password to connect to [rawdb][7].
* **dbserveraddress_output**: This is the default setup of [rawdb][7] running locally.
* **rawdb**: The default database name of [rawdb][7].
* **csv_delimiter**: Flickr YFCC100M data is separated by tabs,
  which is specified in [lbsntransform][3] as `$'\t'` via the command line 
* **file_type**: Flickr YFCC100M data format is CSV (line separated).
* **zip_records**: Length of `yfcc100m_dataset.csv` and `yfcc100m_places.csv` 
  matches. This tells [lbsntransform][3] to concatenate both files on stream read.

**Note:**
Reading the full dataset into the database will require at least 50 GB 
of hard drive and, depending on your hardware, up to several days of processing.
You can read the dataset partially by adding `--transferlimit 10000`, 
to only read the first 10000 entries (e.g.).

[1]: https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
[2]: https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67
[3]: https://lbsn.vgiscience.org/lbsntransform/docs/
[4]: https://lbsn.vgiscience.org/lbsntransform/docs/api/input/mappings/field_mapping_yfcc100m.html#lbsntransform.input.mappings.field_mapping_yfcc100m.FieldMappingYFCC100M
[5]: https://lbsn.vgiscience.org/lbsntransform/docs/api/input/mappings/index.html
[6]: https://gitlab.vgiscience.de/lbsn/tools/full-stack-lbsn
[7]: https://gitlab.vgiscience.de/lbsn/databases/rawdb
[8]: https://gitlab.vgiscience.de/lbsn/databases/hlldb
[9]: https://gitlab.vgiscience.de/lbsn/tools/pgadmin
[10]: https://gitlab.vgiscience.de/lbsn/tools/jupyterlab

[git]: https://git-scm.com/
[docker]: https://www.docker.com/
[WSL]: https://docs.microsoft.com/de-de/windows/wsl/install-win10
[WSL2]: https://devblogs.microsoft.com/commandline/wsl2-will-be-generally-available-in-windows-10-version-2004/
<!-- #endregion -->

# Prepare RAW data for grid aggregation

## Defining the Query

We're using a two database setup:

- **rawdb** refers to the original social media 
  data that is publicly available
- **hlldb** refers to the privacy-aware data 
  collection that is used in the visualization environment

To query original data from **rawdb**, and convert to **hlldb**, 
we're using a [Materialized View][pgdocs-mat-view].

**Materialized View?**
Materialized views are static subsets extracted from larger PostgreSQL tables.
MViews provide a performant filter for large queries. In a real-world example,
data would typically be directly streamed and filtered, without the need for a
Materialized View.
    
[pgdocs-mat-view]: https://www.postgresql.org/docs/12/rules-materializedviews.html


## Connect to rawdb


### Load dependencies

```python
import os, sys
import psycopg2 # Postgres API
import geoviews as gv
import holoviews as hv
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
# Load helper module from ../py/module/tools.py
# this also allows to import code from other
# jupyter notebooks, synced to *.py with jupytext
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"/py")
from modules import tools, preparations
```

Initialize Bokeh and shapely.speedups. Set pandas colwidth.

```python
preparations.init_imports()
pd.set_option('display.max_colwidth', 25)
```

### Establish connection

Password is loaded from `.env` file specified in container setup [hlldb](https://gitlab.vgiscience.de/lbsn/databases/hlldb).

The docker stack contains a full backup of the YFCC database converted to the privacy-aware datastructure. In this Notebook, we're only working with a small part of the data from the table `spatial.latlng`.


Define credentials as environment variables

```python
db_user = "postgres"
db_pass = os.getenv('POSTGRES_PASSWORD')
# set connection variables
db_host = "rawdb"
db_port = "5432"
db_name = "rawdb"
```

Connect to raw database:

```python
db_connection = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass
)
```

Test connection:

```python
db_query = """
    SELECT 1;
"""
# create pandas DataFrame from database data
df = pd.read_sql_query(db_query, db_connection)
display(df.head())
```

Simplify query access:

```python
db_conn = tools.DbConn(db_connection)
db_conn.query("SELECT 1;")
```

If any SQL results in an error, the cursor cannot be used again. In this case, run `db_connection.rollback()` once, to reset the cursor.


### Create Query Schema

Create a new schema called `mviews` and 
update Postgres `search_path`, to include new schema:

```python
sql_query = """
CREATE SCHEMA IF NOT EXISTS mviews;
ALTER DATABASE rawdb
SET search_path = "$user",
                  social,
                  spatial,
                  temporal,
                  topical,
                  interlinkage,
                  extensions,
                  mviews;"""
```

Since the above query will not return any result, we'll directly use the psycopg2 cursor object:

```python
cur = db_connection.cursor()
cur.execute(sql_query)
print(cur.statusmessage)
```

### Prepare query and cryptographic hashing

HyperLogLog uses [MurMurHash], which is a non-cryptographic hashing
algorithm. This opens up some vulnerabilities such as described by 
Desfontaines et al. (2018).

As an additional measurement to prevent re-identification of users
in the final dataset, we are are using Postgres [pgcrypto extension]
to hash any IDs with a secure, unique [key].
        
**References:**

Desfontaines, D., Lochbihler, A., & Basin, D. (2018). Cardinality Estimators do not 
Preserve Privacy. 1–21.

[MurMurHash]: https://en.wikipedia.org/wiki/MurmurHash
[pgcrypto extension]: https://www.postgresql.org/docs/12/pgcrypto.html
[key]: https://en.wikipedia.org/wiki/Salt_(cryptography)


**Create the pgcrypto extension:**

```python
sql_query = "CREATE EXTENSION IF NOT EXISTS pgcrypto SCHEMA extensions;"
cur.execute(sql_query)
print(cur.statusmessage)
```

Prepare cryptographic hash function. The following function
will take an `id` and a `key` (the seed value) 
to produce a new, unique hash that is returned in `hex` encoding.

```python
sql_query = """
/* Produce pseudonymized hash of input id with skey
 * - using skey as seed value
 * - sha256 cryptographic hash function
 * - encode in hex to reduce length of hash
 * - remove trailing '=' from base64 string
 * - return as text
 * - optional: truncate x characters from string
 */
CREATE OR REPLACE FUNCTION 
extensions.crypt_hash (IN id text, IN skey text)
RETURNS text
AS $$
    SELECT
        RTRIM(
            ENCODE(
                HMAC(
                    id::bytea,
                    skey::bytea,
                    'sha256'), 
                'base64'),
            '=')
$$
LANGUAGE SQL
STRICT;
"""
cur.execute(sql_query)
print(cur.statusmessage)
```

**Note:** Cryptographic hashing alone will only produce pseudonymized data,
since any id still relates to a single user (or post, etc.). Therefore,
pseudonymization is considered a weak measure, which can be easily reversed, 
e.g. through [rainbow tables] or context lookup.

It is used here as an additional means to protect HLL sets
from intersection attacks, e.g. as is discussed by
Desfontaines et al. (2018).

**What is a seed value?** The seed value (`skey`) is a secret that is used, together with the encryption 
    function (e.g. sha256) to produce a unique, collision-free output value (the hashed id). 
    The same ID will be converted to a different hash if the seed value is changed. 
    Therefore, in our case, the seed value must remain the same during the entire processing
    of data. In this case, the seed is called a _key_. This key can be destroyed afterwards, 
    if no subsequent updates are necessary.


## Apply RAW query


In this step, several filters are applied to significantly 
reduce data, as a provisional measure to reduce privacy risks.

The topic selection query below the following data reduction steps:

- use only coordinates, post_guids, user_guids and userday
- filter only photos with geoaccuracy 'place', 'latlng' or 'city',
  which broadly equals [Flickr geoaccuracy levels] 8-16
- reduce spatial granularity to about 5km accuracy,
  by using PostGis [GeoHash] function, [ST_GeoHash]
- apply cryptographic hashing to `post_guid`, `user_guid` and `userday`
  using the `crypt_hash` function defined earlier
- reduce temporal granularity of `post_create_date` to `yyyy-MM-dd`, concat with user_guid
    
[Flickr geoaccuracy levels]: https://www.flickr.com/services/api/flickr.photos.geo.photosForLocation.html
[Geohash]: https://en.wikipedia.org/wiki/Geohash
[ST_GeoHash]: https://postgis.net/docs/ST_GeoHash.html


First, set the secret key (CRYPT_KEY) for the crypt function, ideally by an external config file.


Optional cleanup step:

```python
sql_query = f"""
DROP MATERIALIZED VIEW mviews.spatiallatlng_raw_geohash_{GEOHASH_PRECISION:02};
"""
cur.execute(sql_query)
print(cur.statusmessage)
```

```python
%%time
sql_query = f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS mviews.spatiallatlng_raw_geohash_{GEOHASH_PRECISION:02} AS
SELECT  extensions.crypt_hash(t1.post_guid, '{CRYPT_KEY}') as "post_guid",
        ST_Y(ST_PointFromGeoHash(ST_GeoHash(t1.post_latlng, {GEOHASH_PRECISION}), {GEOHASH_PRECISION})) As "latitude", 
        ST_X(ST_PointFromGeoHash(ST_GeoHash(t1.post_latlng, {GEOHASH_PRECISION}), {GEOHASH_PRECISION})) As "longitude", 
        extensions.crypt_hash(t1.user_guid, '{CRYPT_KEY}') as "user_guid",
        extensions.crypt_hash(t1.user_guid || to_char(t1.post_create_date, 'yyyy-MM-dd'), '{CRYPT_KEY}') as "userday",
        t1.post_geoaccuracy
FROM   topical.post t1
WHERE t1.post_geoaccuracy IN ('place', 'latlng', 'city');
"""
cur.execute(sql_query)
print(cur.statusmessage)
```

Info: **GeoHash?**
    
A **GeoHash of `5`** means, coordinates are reduced to 5 decimal digits 
maximum length of lat/lng. Compare the following table:
    
| Precision (number of digits) | Distance of Adjacent Cell in Meters |
|------------------------------|-------------------------------------|
| 1                            | 5003530                             |
| 2                            | 625441                              |
| 3                            | 123264                              |
| 4                            | 19545                               |
| 5                            | 3803                                |
| 6                            | 610                                 |
| 7                            | 118                                 |
| 8                            | 19                                  |
| 9                            | 3.71                                |
| 10                           | 0.6                                 |

**Table:** GeoHash length and corresponding geoaccuracy in meters (Source: [Wikipedia][1]).
    
[1]: https://en.wikipedia.org/wiki/Geohash


# Convert data from rawdb to hlldb  


First, create a connection from hlldb to rawdb.


## Prepare rawdb

On rawdb, create an `lbsn_reader` with read-only privileges for schema `mviews`.

```python
sql_query = """
SELECT 1 FROM pg_roles WHERE rolname='lbsn_reader'
"""
result = db_conn.query(sql_query)
if result.empty:
    # if user does not exist
    sql_query = f"""    
    CREATE USER lbsn_reader WITH
        LOGIN
        INHERIT
        PASSWORD '{USER_KEY}';
        
    GRANT CONNECT ON DATABASE rawdb TO lbsn_reader;
    GRANT USAGE ON SCHEMA mviews TO lbsn_reader;
    ALTER DEFAULT PRIVILEGES IN SCHEMA mviews GRANT SELECT ON TABLES TO lbsn_reader;
    """
    cur.execute(sql_query)
    print(cur.statusmessage)
```

## Connect hlldb to rawdb

By using Foreign Table, this step will establish the connection 
between hlldb to rawdb.


On hlldb, install [postgres_fdw extension]:
    
[postgres_fdw extension]: https://www.postgresql.org/docs/12/postgres-fdw.html

```python
sql_query = """
CREATE EXTENSION IF NOT EXISTS postgres_fdw SCHEMA extensions;
"""
cur.execute(sql_query)
print(cur.statusmessage)
```

Create Foreign Server on hlldb:

```python
db_connection_hll = psycopg2.connect(
        host="hlldb",
        port=db_port,
        dbname="hlldb",
        user=db_user,
        password=db_pass
)
db_conn_hll = tools.DbConn(db_connection_hll)
cur_hll = db_connection_hll.cursor()
cur_hll.execute("SELECT 1;")
print(cur_hll.statusmessage)
```

```python
sql_query = f"""
CREATE SERVER IF NOT EXISTS lbsnraw 
FOREIGN DATA WRAPPER postgres_fdw
OPTIONS (
    host 'rawdb',
    dbname 'rawdb',
    port '5432',
    keepalives '1',
    keepalives_idle '30',
    keepalives_interval '10',
    keepalives_count '5',
    fetch_size '500000');
CREATE USER MAPPING IF NOT EXISTS for postgres
    SERVER lbsnraw 
    OPTIONS (user 'lbsn_reader', password '{USER_KEY}');
"""
cur_hll.execute(sql_query)
print(cur_hll.statusmessage)
```

Import foreign table definition on the hlldb.

```python
sql_query = f"""
SELECT EXISTS (
   SELECT FROM information_schema.tables 
   WHERE  table_schema = 'extensions'
   AND    table_name   = 'spatiallatlng_raw_geohash_{GEOHASH_PRECISION:02}'
   );
"""
result = db_conn_hll.query(sql_query)
if not result["exists"][0]:
    # only import table 
    # if it hasn't been imported already
    sql_query = f"""
    IMPORT FOREIGN SCHEMA mviews
        LIMIT TO (spatiallatlng_raw_geohash_{GEOHASH_PRECISION:02})
        FROM SERVER lbsnraw 
        INTO extensions;
    """
    cur_hll.execute(sql_query)
    print(cur_hll.statusmessage)
```

<!-- #region -->
**Tip:** Optionally: optimize chunk size
        
Depending on your hardware, optimizing Postgres 
`fetch_size` may increase processing speed:
```sql
ALTER SERVER lbsnraw
OPTIONS (SET fetch_size '50000');
```
<!-- #endregion -->

## Prepare conversion of raw data to hll

We're going to use `spatial.latlng` from the [HLL Structure](https://gitlab.vgiscience.de/lbsn/structure/hlldb/-/blob/master/structure/98-create-tables.sql#L160) 
definition. The structure for this table is already available,
by default, in hlldb.

```python
sql_query = """
CREATE TABLE IF NOT EXISTS spatial.latlng (
    latitude float,
    longitude float,
    PRIMARY KEY (latitude, longitude),
    latlng_geom geometry(Point, 4326) NOT NULL)
INHERITS (
    social.user_hll, -- e.g. number of users/latlng (=upl)
    topical.post_hll, -- e.g. number of posts/latlng
    temporal.date_hll -- e.g. number of dates/latlng
);
"""
cur_hll.execute(sql_query)
print(cur_hll.statusmessage)
```

## HyperLogLog parameters

The HyperLogLog extension for Postgres from [Citus] that we're using here,
contains several tweaks, to optimize performance, that can affect sensitivity of data.

From a privacy perspective, for example,
it is recommended to disable [explicit mode].

**Explicit mode?** When explicit mode is active, full IDs will be stored
    for small sets. In our case, any coordinates frequented
    by few users (outliers) would store full user and post IDs.
    
[Citus]: https://github.com/citusdata/postgresql-hll
[explicit mode]: https://github.com/citusdata/postgresql-hll/blob/master/REFERENCE.md#metadata-functions


To disable explicit mode:

```python
sql_query = """
SELECT hll_set_defaults(11, 5, 0, 1);
"""
db_conn_hll.query(sql_query)
```

<!-- #region -->
From now on, HLL sets  will directly be promoted to sparse.

<div class="alert alert-warning">
    
In Sparse Hll Mode, more data is stored than in Full Hll Mode, 
as a means to improve accuracy for small sets. As pointed out by 
Desfontaines et al. 2018, this may make
re-identification easier. Optionally disable sparse mode:

```sql
SELECT hll_set_defaults(11,5, 0, 0);
```

The caveat here is that both required storage size and
processing time increase.
    
</div>
<!-- #endregion -->

## Aggregation step: Convert data to Hll


This is the actual data collection and aggregation step. In the query below,
different metrics are collected that are typical
for LBSM visual analytics (postcount, usercount, userdays).


Make sure that no data exists in table spatial.latlng. Optional cleanup step:

```python
sql_query = """
TRUNCATE spatial.latlng;
"""
cur_hll.execute(sql_query)
print(cur_hll.statusmessage)
```

```python
sql_query = """
SELECT * from spatial.latlng limit 10;
"""
results = db_conn_hll.query(sql_query)
results.empty
```

To test: uncomment `--LIMIT 100` below

```python
%%time
if results.empty:
    sql_query = f"""
    INSERT INTO spatial.latlng(
                latitude, 
                longitude, 
                user_hll, 
                post_hll, 
                date_hll, 
                latlng_geom)
        SELECT  latitude,
                longitude,
                hll_add_agg(hll_hash_text(user_guid)) as user_hll,
                hll_add_agg(hll_hash_text(post_guid)) as post_hll,
                hll_add_agg(hll_hash_text(userday)) as date_hll,
                ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) as latlng_geom
        FROM extensions.spatiallatlng_raw_geohash_{GEOHASH_PRECISION:02}
        --LIMIT 100
        GROUP BY latitude, longitude;
    """
    cur_hll.execute(sql_query)
    print(cur_hll.statusmessage)
else:
    print("Table contains already data. Please cleanup first (TRUNCATE spatial.latlng;)")
```

Commit changes to db and close connection:

```python
db_connection_hll.commit()
```

```python
db_connection_hll.close ()
```

<!-- #region -->
This query will take some time, depending on your machine.

If, for some reason, the query does not work - try directly connecting to the psql console:

```bash
docker exec -it lbsn-hlldb /bin/bash
psql -h localhost -p 5432 -U postgres hlldb
```
<!-- #endregion -->

**Convert notebook to HTML**

```python
!jupyter nbconvert --to html_toc \
    --output-dir=../out/html ./01_preparations.ipynb \
    --template=../nbconvert.tpl \
    --ExtractOutputPreprocessor.enabled=False # create single output file
```

# Visualize data

In the raw and hll notebook, we'll aggregate data per 100km grid. Follow in [02_yfcc_gridagg_raw.ipynb](02_yfcc_gridagg_raw.html) and [03_yfcc_gridagg_hll.ipynb](03_yfcc_gridagg_hll.html)

```python

```
