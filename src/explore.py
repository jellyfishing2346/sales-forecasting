import pandas as pd
import matplotlib.pyplot as plt

def basic_eda(df):
    print('Data shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('Missing values:', df.isnull().sum())
    print(df.describe())
    df.hist(figsize=(10,8))
    plt.tight_layout()
    plt.show()
