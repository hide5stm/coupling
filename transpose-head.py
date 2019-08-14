# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
from itertools import zip_longest


def read_head(f):
    cols = []
    for line in f:
        for col in line.strip().split(','):
            cols.append(col.strip())
        break
    return cols

def print_heads(heads, max_col_len):
    for xcols in list(zip_longest(*heads, fillvalue='--')):
        print('  '.join([f'{x:<{max_col_len}}' for x in xcols]))


def main(paths):
    heads = []
    max_col_len = 0
    for path in paths:
        with open(path, 'r') as f:
            cols = [path]
            cols += read_head(f)
            max_col_len = max(max_col_len, *[len(col) for col in cols])
            heads.append(cols)


    print_heads(heads, max_col_len)
            

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
