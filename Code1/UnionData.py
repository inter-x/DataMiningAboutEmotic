import argparse
import os
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

'''
person、Kendall、spearman
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='C:/Users/Daisy/Python_code/Emotions/pythonProject2/Data/emotic_pre',
                        help='Path to preprocessed data npy files/ csv files')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Load data preprocessed npy files
    test_context = np.load(os.path.join(args.data_path, 'test_cat_arr.npy'))
    train_context = np.load(os.path.join(args.data_path, 'train_cat_arr.npy'))
    val_context = np.load(os.path.join(args.data_path, 'val_cat_arr.npy'))
    temp = [test_context, train_context, val_context]

    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy',
           'Yearning']
    df_test = pd.DataFrame(test_context)
    df_train = pd.DataFrame(train_context)
    df_val = pd.DataFrame(val_context)
    frame = [df_test, df_train,df_val]
    df = pd.concat(frame)
    df.columns=cat
    #np.set_printoptions(threshold=np.inf)
    print(df)

    coe1 = df.corr()
    coe2 = df.corr('kendall')
    coe3 = df.corr('spearman')
    sns.heatmap(coe1, cmap='Blues', annot=True)
    plt.show()
    sns.heatmap(coe2, cmap='Blues', annot=True)
    plt.show()
    sns.heatmap(coe3, cmap='Blues', annot=True)
    plt.show()
    print(coe1)




