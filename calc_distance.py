# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
import numpy as np
import csv


def load(file):
    data = []
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i < 2: # 先頭2行はskip
                continue
            data.append(line.strip().split())

    return data

def validate(data):
    vs = []
    for d in data:
        x, y, z = [float(x) for x in d[1:]]
        vs.append((d[0], x, x, z))

    return vs

def distance(xyz0, xyz1):
    v0 = np.array(xyz0)
    v1 = np.array(xyz1)

    return (v1 - v0).tolist(), np.linalg.norm(v1 - v0)

def calc(vs0, vs1):
    i, atom0 = vs0[0], vs0[1]
    j, atom1 = vs1[0], vs1[1]
    d = distance(vs0[2:], vs1[2:])

    (x, y, z), d = d
    return i, j, atom0, atom1, x, y, z, d

def make_feature(data):
    ret = []
    for vs0, vs1 in combinations(data, 2):
        row = calc(vs0, vs1)
        ret.append(row)

    return ret

def add_index(data):
    ret = []
    for i, v in enumerate(data):
        vs = [i]
        vs += v
        ret.append(vs)

    return ret

def save_csv(file, rows, header):
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def main(paths):
    in_dir, out_dir = paths

    for file in tqdm(Path(in_dir).glob('*.xyz')):
        data = load(file)
        data = validate(data)
        data = add_index(data)
        features = make_feature(data)

        out_path = Path(f'{out_dir}/dist_{file.stem}.csv')
        header = ['index0', 'index1', 'atom0', 'atom1', 'diff-x', 'diff-y', 'diff-z', 'dist']
        save_csv(out_path, features, header)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
