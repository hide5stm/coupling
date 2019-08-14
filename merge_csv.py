# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

'''
index,n1_index,n1_atom,n1_dist,n2_index,n2_atom,n2_dist,n3_index,n3_atom,n3_dist,n4_index,n4_atom,n4_dist,n5_index,n5_atom,n5_dist,nearest_C_dist,nearest_O_dist,nearest_N_dist,nearest_F_dist,nearest_H_dist,molecule_name
'''

def merge_df(df, struct):
    df = pd.merge(df, struct,
                  left_on=['molecule_name', 'atom_index_0'],
                  right_on=['molecule_name', 'index'],
                  how='left')
    df.drop('index', axis=1, inplace=True)
    df.rename(columns={'atom': 'atom0'}, inplace=True)
    assert size == len(df)
    df = pd.merge(df, struct,
                  left_on=['molecule_name', 'atom_index_1'],
                  right_on=['molecule_name', 'atom_index'],
                  how='left')
    df.drop('atom_index', axis=1, inplace=True)
    df.rename(columns={'atom': 'atom1'}, inplace=True)
    assert size == len(df)
    return df


def main():
    in_file1, in_file2, out_file = paths
    # temp/atom.csv temp/atom-new.csv
    # 'temp/atom-features.csv'
    df = 
    
    df = merge_df()

    df0.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
