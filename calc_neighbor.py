# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
import numpy as np
import csv
from collections import defaultdict, Iterable
from operator import itemgetter, attrgetter
import pandas as pd


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def dict_collect(data):
    '''
    index0,index1,atom0,atom1,diff-x,diff-y,diff-z,dist
    index0の原子毎に原子間距離順にソート
    '''
    dic = defaultdict(list)
    for row in data:
        # index1, atom1, dist
        dic[row[0]].append((row[1], row[3], row[-1]))
        # 逆方向
        # index0, atom0, dist
        dic[row[1]].append((row[0], row[2], row[-1]))

    return dic

def sort_neighbor(dic):
    ''' 近い順にsort '''
    for vs in dic.values():
        vs.sort(key=itemgetter(2), reverse=False)

def select_feature(dic):
    '''
    dic: 原子のindexをkeyにしたdict
    (idx, (atom0, index1, atom1, dist)) : 1番目、2番目、3番目、4番目、5番目
    '''
    lines = []
    for idx, vs in dic.items():
        xs = vs[0:5] + [(None, None, None)] * (5 - len(vs[0:5]))
        assert len(xs) == 5
        lines.append((idx, xs))

    return lines


def pick_nearest(dic, atom):
    lines = []
    for vs in dic.values():
        line = None
        for _idx1, atom1, dist in vs:
            if atom1 == atom:
                line = [dist]
                break

        if line is None:
            line = [None]

        lines.append(line)

    return lines

def join_nearest(lines, dic, atom):
    li = pick_nearest(dic, atom)
    assert len(lines) == len(li)
    lines = list(zip(lines, li))
    return lines


def make_feature(data):
    ''' data: 分子内の原子情報 '''
    dic = dict_collect(data)
    sort_neighbor(dic)
    lines = select_feature(dic)
    lines = join_nearest(lines, dic, 'C')
    lines = join_nearest(lines, dic, 'O')
    lines = join_nearest(lines, dic, 'N')
    lines = join_nearest(lines, dic, 'F')
    lines = join_nearest(lines, dic, 'H')

    return lines

def main(paths):
    in_dir, out_dir = paths
    cols = ['index',
            'n1_index', 'n1_atom', 'n1_dist',
            'n2_index', 'n2_atom', 'n2_dist',
            'n3_index', 'n3_atom', 'n3_dist',
            'n4_index', 'n4_atom', 'n4_dist',
            'n5_index', 'n5_atom', 'n5_dist',
            'nearest_C_dist',
            'nearest_O_dist',
            'nearest_N_dist',
            'nearest_F_dist',
            'nearest_H_dist',
    ]

    for file in tqdm(Path(in_dir).glob('dist_*.csv')):
        df = pd.read_csv(file)
        data = df.values.tolist()
        features = make_feature(data)
        features = [flatten(row) for row in features]
        df = pd.DataFrame(features, columns=cols)

        base = file.stem.replace('dist_', '')
        out_path = Path(f'{out_dir}/feature_{base}.csv')
        df.to_csv(out_path, index=False)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
