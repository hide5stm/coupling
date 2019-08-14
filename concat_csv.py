# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd


def main(paths):
    in_dir, out_dir = paths
    prefix = 'feature_'
    fname = f'{prefix}*.csv'

    for file in tqdm(Path(in_dir).glob(fname)):
        df = pd.read_csv(file)
        mol_name = file.stem.replace(prefix, '')
        df['molecule_name'] = mol_name
        df.to_csv(f'{out_dir}/{mol_name}.csv', index=False)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
