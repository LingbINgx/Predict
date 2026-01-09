

import pandas as pd
import numpy as np


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data






def main():
    data = read_data('./ai4i2020.csv')
    print(data.head())
    print(len(data))
    


if __name__ == "__main__":
    main()
