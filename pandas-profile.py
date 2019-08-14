import pathlib
import pandas as pd
import numpy as np
import pandas_profiling as pdp



def main():
    path = pathlib.Path('input')
    for csvfile in path.glob('train.csv'):
        print(str(csvfile))
        try:
            data = pd.read_csv(csvfile)
            profile = pdp.ProfileReport(data)
            profile.to_file('{}.html'.format(csvfile))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise



if __name__ == '__main__':
    main()
