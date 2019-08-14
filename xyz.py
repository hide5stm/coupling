
import pandas as pd
import numpy as np


def main():
    struct = pd.read_csv('./input/structures.csv')

    '''
    for i in range(1, 133885+1):
        print('dsgdb9nsd_{:06}'.format(i))
    '''

    names = struct.groupby('molecule_name')
    moles = {}
    for name in names.groups:
        moles[name] = names.get_group(name)

    print(len(moles))





if __name__ == '__main__':
    main()
