import sqlite3
from sqlite3 import Error
from astropy.table import Table
import numpy as np
import math as mh
import os
from pathlib import Path
from tqdm import tqdm

col_heads = [i.strip('[').strip(']') 
             for i in np.genfromtxt(os.path.join(Path.home(),'.prose/tic_column_description.txt'), 
                                    unpack=True, dtype='str')[0]]

#CHANGE THIS TO ABSOLUTE PATH - to do!
dbfile='/Volumes/TOSHIBA EXT/TIC_new.db'

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

table_list=[
    f'TIC_{20+i*2}_{22+i*2}' for i in range(35)
]

def search_db(minra,maxra,mindec,maxdec,dec):
    dec_range=[f'{abs(mh.floor(dec))-1}', f'{abs(mh.floor(dec))}', f'{abs(mh.floor(dec))+1}']
    tables2check=[i for i in table_list if dec_range[0] in i]+[i for i in table_list if dec_range[1] in i]+ \
    [i for i in table_list if dec_range[2] in i]
    tables2check=np.unique(np.array(tables2check)).tolist()
    output=[]
    conn=create_connection(dbfile)
    cur=conn.cursor()

    for i in tables2check:
        cur.execute(f"SELECT * FROM {i} WHERE ra<=? and ra>=? and dec<=? and dec>=?", 
                (maxra,minra,maxdec,mindec,))
        for j in tqdm(cur.fetchall()):
            output.append(j)
        #output+=cur.fetchall()

    cur.close()
    output=[[np.ma.masked if i== None else i for i in j ] for j in output]
    table=Table(names=(col_heads), rows=output)

    return table