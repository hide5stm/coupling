# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

import const


pd.set_option('max_columns',20)
pd.set_option('max_rows',100)
pd.set_option('expand_frame_repr', False)
pd.options.display.float_format = '{:,.3f}'.format


answer_col = 'scalar_coupling_constant'
type_col = 'type'

def merge_struct(df, struct):
    size = len(df)
    df = pd.merge(df, struct,
                  left_on=['molecule_name', 'atom_index_0'],
                  right_on=['molecule_name', 'atom_index'],
                  how='left')
    df.drop('atom_index', axis=1, inplace=True)
    df.rename(columns={'x': 'x0', 'y': 'y0', 'z': 'z0'}, inplace=True)
    df.rename(columns={'atom': 'atom0'}, inplace=True)
    assert size == len(df)
    df = pd.merge(df, struct,
                  left_on=['molecule_name', 'atom_index_1'],
                  right_on=['molecule_name', 'atom_index'],
                  how='left')
    df.drop('atom_index', axis=1, inplace=True)
    df.rename(columns={'x': 'x1', 'y': 'y1', 'z': 'z1'}, inplace=True)
    df.rename(columns={'atom': 'atom1'}, inplace=True)
    assert size == len(df)

    print(df.columns)
    return df

def merge_feature(df, fea):
    df = pd.merge(df, fea,
                  left_on=['molecule_name', 'atom_index'],
                  right_on=['molecule_name', 'atom_index'],
                  how='left')

    return df
    

def append_neibor(nearests, nei, i):
    for j in range(len(nei)):
        nearests[f'near_dist_{j}'].append(nei[f'dist_{i}'].iloc[j])
        nearests[f'near_atom_{j}'].append(nei[f'atom'].iloc[j])            
        nearests[f'near_x_{j}'].append(nei[f'x_{i}'].iloc[j])
        nearests[f'near_y_{j}'].append(nei[f'y_{i}'].iloc[j])
        nearests[f'near_z_{j}'].append(nei[f'z_{i}'].iloc[j])

    # 最寄りが5より小さければ、足りないデータはNoneで埋める
    for j in range(5-len(nei)+1, 5):
        nearests[f'near_dist_{j}'].append(None)
        nearests[f'near_atom_{j}'].append(None)
        nearests[f'near_x_{j}'].append(None)
        nearests[f'near_y_{j}'].append(None)
        nearests[f'near_z_{j}'].append(None)
            

def nearest_atom(df, name, i):
    assert name in ('C', 'O', 'N', 'F', 'H')
    rows = df.loc[(df['atom'] == name) & (df['atom_idx'] != i)]
    cols = [f'dist_{i}', f'x_{i}', f'y_{i}', f'z_{i}']
    if not rows.empty:
        row = rows.iloc[0]
        row = row[cols]
        return (row[f'dist_{i}'], row[f'x_{i}'], row[f'y_{i}'], row[f'z_{i}'])
    else:
        return tuple([None] * len(cols))

def append_atom(d, name, row):
    d[f'near_dist_{name}'].append(row[0])
    d[f'near_x_{name}'].append(row[1])
    d[f'near_y_{name}'].append(row[2])
    d[f'near_z_{name}'].append(row[3])
    
def create_features_mol(df_mols):
    ''' 分子構造から特徴量を作成
        原子毎の近傍原子5個、原子の種類毎に最寄りの原子を1つ探して、
        dfを作成して返す
    '''
    print('create feature by molecule')
    nearests = {
        'molecule_name': [], 'atom_index': [],
        'near_atom_0': [], 'near_atom_1': [], 'near_atom_2': [], 'near_atom_3': [], 'near_atom_4': [],
        'near_dist_0': [], 'near_dist_1': [], 'near_dist_2': [], 'near_dist_3': [], 'near_dist_4': [],
        'near_x_0': [], 'near_x_1': [], 'near_x_2': [], 'near_x_3': [], 'near_x_4': [],
        'near_y_0': [], 'near_y_1': [], 'near_y_2': [], 'near_y_3': [], 'near_y_4': [],
        'near_z_0': [], 'near_z_1': [], 'near_z_2': [], 'near_z_3': [], 'near_z_4': [],
        'near_dist_C': [], 'near_x_C': [], 'near_y_C': [], 'near_z_C': [],
        'near_dist_O': [], 'near_x_O': [], 'near_y_O': [], 'near_z_O': [],
        'near_dist_N': [], 'near_x_N': [], 'near_y_N': [], 'near_z_N': [],
        'near_dist_F': [], 'near_x_F': [], 'near_y_F': [], 'near_z_F': [],
        'near_dist_H': [], 'near_x_H': [], 'near_y_H': [], 'near_z_H': [],
    }

    for name, mol in tqdm(df_mols.groupby('molecule_name')):
        '''
        if name != 'dsgdb9nsd_000008':
            continue
        '''
        mol.drop(columns=['molecule_name'], axis=1, inplace=True)
        vs = pd.DataFrame({'atom_idx': mol['atom_index'], 'atom': mol['atom']})
        for atom0 in mol.itertuples():
            i = atom0.atom_index
            vs[f'x_{i}'] = (mol['x'] - atom0.x).values
            vs[f'y_{i}'] = (mol['y'] - atom0.y).values
            vs[f'z_{i}'] = (mol['z'] - atom0.z).values
            vs[f'dist_{i}'] = np.sqrt(vs[f'x_{i}'] ** 2
                                      + vs[f'y_{i}'] ** 2
                                      + vs[f'z_{i}'] ** 2).values


        for i in range(len(mol)):
            sorted = vs.sort_values(by=[f'dist_{i}'], ascending=True)
            # 近い順に5つ
            nei = sorted[1:6][['atom_idx', 'atom', f'dist_{i}', f'x_{i}', f'y_{i}', f'z_{i}']]
            #print(i, nei)
            nearests['molecule_name'].append(name)
            nearests['atom_index'].append(i)
            append_neibor(nearests, nei, i)

            # 最寄りのC, O, N, F, H
            append_atom(nearests, 'C', nearest_atom(sorted, 'C', i))
            append_atom(nearests, 'O', nearest_atom(sorted, 'O', i))
            append_atom(nearests, 'N', nearest_atom(sorted, 'N', i))
            append_atom(nearests, 'F', nearest_atom(sorted, 'F', i))
            append_atom(nearests, 'H', nearest_atom(sorted, 'H', i))

        '''
        break
        '''

    nearests = pd.DataFrame(nearests)
    print(nearests)
    return nearests


def create_features(df):
    ''' 特徴量の作成 '''
    # 原素記号を原子量に変換
    print('apply atom weght')
    df['atom_w_0'] = df['atom0'].progress_apply(lambda x: const.atomic_weight[x])
    df['atom_w_1'] = df['atom1'].progress_apply(lambda x: const.atomic_weight[x])
    print('apply atom No.')
    df['atom_no_0'] = df['atom0'].progress_apply(lambda x: const.atomic_no[x])
    df['atom_no_1'] = df['atom1'].progress_apply(lambda x: const.atomic_no[x])

    # 原子0と1の位置座標を xyz座標から極座標に変換
    df['x'] = df['x1'] - df['x0']
    df['y'] = df['y1'] - df['y0']
    df['z'] = df['z1'] - df['z0']
    df.drop(columns=['x0', 'x1', 'y0', 'y1', 'z0', 'z1'], axis=1, inplace=True)

    df['r'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
    df['theta'] = np.arccos(df['z'] / df['r'])
    df['phi'] = np.arccos(df['x'] / np.sqrt(df['x'] ** 2 + df['y'] ** 2))
    df.drop(columns=['x', 'y', 'z'], axis=1, inplace=True)

    return df


def load_data():
    base = './input'
    # df_pot_energy = pd.read_csv(f'{base}/potential_energy.csv')
    # df_mul_charges = pd.read_csv(f'{base}/mulliken_charges.csv')
    # df_scal_coup_contrib = pd.read_csv(f'{base}/scalar_coupling_contributions.csv')
    # df_magn_shield_tensor = pd.read_csv(f'{base}/magnetic_shielding_tensors.csv')
    # df_dipole_moment = pd.read_csv(f'{base}/dipole_moments.csv')
    df_structure = pd.read_csv(f'{base}/structures.csv')
    df_train = pd.read_csv(f'{base}/train.csv')
    df_test = pd.read_csv(f'{base}/test.csv')

    # 分子構造から特徴量を作成
    df = create_features_mol(df_structure)
    df_structure = merge_feature(df_structure, df)
    df_structure.to_csv('./temp/feature_struct.csv', index=False)

    if True:
        return

    # TODO: まずtype1種類でためす
    df_train = df_train.groupby('type').get_group('1JHC')
    df_test = df_test.groupby('type').get_group('1JHC')
    
    # 構造データをマージ
    df_train = merge_struct(df_train, df_structure)
    df_test = merge_struct(df_test, df_structure)

    df_train = create_features(df_train)
    df_test = create_features(df_test)

    return df_train, df_test


def main():
    df_train, df_test = load_data()

    return

    out_path = './temp'
    df_train.to_csv(f'{out_path}/train-new.csv', index=False)
    df_test.to_csv(f'{out_path}/test-new.csv', index=False)


if __name__ == '__main__':
    main()
