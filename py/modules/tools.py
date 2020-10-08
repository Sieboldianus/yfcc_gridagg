"""GridAgg Notebook Tools"""

import csv
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path
from collections import namedtuple
from IPython.display import display

class DbConn(object):

    def __init__(self, db_conn):
        self.db_conn = db_conn

    def query(self, sql_query: str) -> pd.DataFrame:
        """Execute Calculation SQL Query with pandas"""
        # create pandas DataFrame from database data
        return pd.read_sql_query(sql_query, self.db_conn)

    def close(self):
        self.db_conn.close()
        
def get_stats_query(table: str):
    """Returns PostgresSQL table count and size stats query
    by Erwin Brandstetter, source:
    https://dba.stackexchange.com/a/23933/139107
    """
    db_query = f"""
    SELECT l.metric, l.nr AS "bytes/ct"
         , CASE WHEN is_size THEN pg_size_pretty(nr) END AS bytes_pretty
         , CASE WHEN is_size THEN nr / NULLIF(x.ct, 0) END AS bytes_per_row
    FROM  (
       SELECT min(tableoid)        AS tbl      -- = 'public.tbl'::regclass::oid
            , count(*)             AS ct
            , sum(length(t::text)) AS txt_len  -- length in characters
       FROM   {table} t
       ) x
     , LATERAL (
       VALUES
          (true , 'core_relation_size'               , pg_relation_size(tbl))
        , (true , 'visibility_map'                   , pg_relation_size(tbl, 'vm'))
        , (true , 'free_space_map'                   , pg_relation_size(tbl, 'fsm'))
        , (true , 'table_size_incl_toast'            , pg_table_size(tbl))
        , (true , 'indexes_size'                     , pg_indexes_size(tbl))
        , (true , 'total_size_incl_toast_and_indexes', pg_total_relation_size(tbl))
        , (true , 'live_rows_in_text_representation' , txt_len)
        , (false, '------------------------------'   , NULL)
        , (false, 'row_count'                        , ct)
        , (false, 'live_tuples'                      , pg_stat_get_live_tuples(tbl))
        , (false, 'dead_tuples'                      , pg_stat_get_dead_tuples(tbl))
       ) l(is_size, metric, nr);
    """
    return db_query

FileStat = namedtuple('File_stat', 'name, size, records')

def get_file_stats(name: str, file: Path) -> Tuple[str, str, str]:
    """Get number of records and size of CSV file"""
    num_lines = f'{sum(1 for line in open(file)):,}'
    size = file.stat().st_size
    size_gb = f'{size/(1024*1024):.2f} MB'
    return FileStat(name, size_gb, num_lines)

def display_file_stats(filelist: Dict[str, Path]):
    """Display CSV """
    df = pd.DataFrame(
        data=[
            get_file_stats(name, file) for name, file in filelist.items()
            ]).transpose()
    header = df.iloc[0]
    df = df[1:]
    df.columns = header
    display(df.style.background_gradient(cmap='viridis'))
    
HllRecord = namedtuple('Hll_record', 'lat, lng, user_hll, post_hll, date_hll')

def strip_item(item, strip: bool):
    if not strip:
        return item
    if len(item) > 120:
        item = item[:120] + '..'
    return item

def get_hll_record(record, strip: bool = None):
    """Concatenate topic info from post columns"""
        
    lat = record.get('latitude')
    lng = record.get('longitude')
    user_hll = strip_item(record.get('user_hll'), strip)
    post_hll = strip_item(record.get('post_hll'), strip)
    date_hll = strip_item(record.get('date_hll'), strip)            
    return HllRecord(lat, lng, user_hll, post_hll, date_hll)

def record_preview_hll(file: Path, num: int = 2):
    """Get record preview for hll data"""
    with open(file, 'r', encoding="utf-8") as file_handle:
        post_reader = csv.DictReader(
                    file_handle,
                    delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL)
        for ix, hll_record in enumerate(post_reader):
            hll_record = get_hll_record(hll_record, strip=True)
            # convert to df for display
            display(pd.DataFrame(data=[hll_record]).rename_axis(
                f"Record {ix}", axis=1).transpose().style.background_gradient(cmap='viridis'))
            # stop iteration after x records
            if ix >= num:
                break