import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


'''

```
$ wc -l input/*.csv
    85004 input/dipole_moments.csv    分子毎の双極子モーメント
    85004 input/potential_energy.csv  分子毎の位置エネルギー
  1533538 input/magnetic_shielding_tensors.csv 原子毎の化学シフトテンソル
  1533538 input/mulliken_charges.csv  原子毎のマリケン電気陰性度(kJ/mol)
  2358658 input/structures.csv        原子毎の構造データ(全130775分子)
  4658148 input/scalar_coupling_contributions.csv  原子間のtype,fc,sd,pso,dso
  4658148 input/train.csv             原子間のカップリング係数
  2505543 input/test.csv              求めるべき原子間のカップリング係数
```

* テスト分子数   45772
* テスト原子数
* トレーニング分子数   85003
* トレーニング原子数 1533538

* テスト原子間数       2505542
* トレーニング原子間数 4658147
'''

def build_molecule_df():
    ''' 分子単位のデータ '''
    mole_dipole = pd.read_csv('./input/dipole_moments.csv')
    mole_enegy  = pd.read_csv('./input/potential_energy.csv')

    molecule = pd.merge(mole_dipole, mole_enegy, on=['molecule_name'])

    print(len(molecule), molecule.head())
    return molecule


def build_atom_df():
    ''' 原子単位のデータ '''
    atom_shield = pd.read_csv('./input/magnetic_shielding_tensors.csv')
    atom_charge = pd.read_csv('./input/mulliken_charges.csv')
    atom_struct = pd.read_csv('./input/structures.csv')

    atom = pd.merge(atom_struct, atom_shield,
                    on=['molecule_name', 'atom_index'],
                    how='left')

    atom = pd.merge(atom, atom_charge,
                    on=['molecule_name', 'atom_index'],
                    how='left')

    print(len(atom), atom.head())
    return atom


def build_couple_df():
    ''' 原子間結合単位のデータ '''

    coupling = pd.read_csv('./input/scalar_coupling_contributions.csv')
    train = pd.read_csv('./input/train.csv')
    #test = pd.read_csv('./input/test.csv')

    coupling = pd.merge(train, coupling,
                    on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                    how='left')

    return coupling


def main():
    mols = build_molecule_df()
    atom = build_atom_df()
    coupling = build_couple_df()

    mols.to_csv('./temp/molecule.csv')
    atom.to_csv('./temp/atom.csv')
    coupling.to_csv('./temp/couping.csv')



if __name__ == '__main__':
    main()
