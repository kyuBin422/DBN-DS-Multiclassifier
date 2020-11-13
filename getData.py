# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   date：          6/1/2020
-------------------------------------------------
   Change Activity:
                   6/1/2020:
-------------------------------------------------
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def getData():
    path = "data.xlsx"
    try:
        X = np.load("achieve/data.npy", allow_pickle=True)
    except:
        X = pd.read_excel(path)
        X = X.values
    # drop time internal data
    X = X[:, 3:].astype(np.float)
    # initial figure
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.scatter(X[:, 0], X[:, 1], label="Original Data", rasterized=True)
    ax1.set_xlabel("wind speed m/s")
    ax1.set_ylabel("grid power Kw/h")
    ax1.legend(loc="upper right", prop={'size': 6})
    # ax1.tight_layout()
    ax1.set_xlim([-1, 35])
    # filter the data by 3 sigma
    X_3_sigma = X.copy()
    for i in range(X.shape[1]):
        lower_range = np.mean(X_3_sigma[:, i]) - 3 * np.std(X_3_sigma[:, i])
        upper_range = np.mean(X_3_sigma[:, i]) + 3 * np.std(X_3_sigma[:, i])
        flag = (X_3_sigma[:, i] > lower_range) & (X_3_sigma[:, i] < upper_range)
        X_3_sigma = X_3_sigma[flag, :]
    ax2.scatter(X_3_sigma[:, 0], X_3_sigma[:, 1], label="Filtered Data by 3-Sigma", rasterized=True)
    ax2.set_xlabel("wind speed m/s")
    ax2.set_ylabel("grid power Kw/h")
    ax2.legend(loc="upper right", prop={'size': 6})
    ax2.set_xlim([-1, 35])
    # ax2.tight_layout()
    del X_3_sigma
    # calculate the interquartile to delete data
    for i in range(X.shape[1]):
        lower_range, upper_range = outlier_treatment(X[:, i])
        flag = (X[:, i] > lower_range) & (X[:, i] < upper_range)
        X = X[flag, :]
    # filter the data by quartiles
    ax3.scatter(X[:, 0], X[:, 1], label="Filtered Data by quartiles", rasterized=True)
    ax3.set_xlabel("wind speed m/s")
    ax3.set_ylabel("grid power Kw/h")
    ax3.legend(loc="upper right", prop={'size': 6})
    ax3.set_xlim([-1, 35])
    # ax3.tight_layout()
    plt.tight_layout()
    plt.savefig("image/3sigma.svg", format='svg', dpi=1600)
    plt.show()
    np.save("achieve/ProcessedData.npy", X)
    return X


def outlier_treatment(data_rows):
    data_rows = np.sort(data_rows)
    Q1, Q3 = np.percentile(data_rows, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


# 954683
# CCE5FF
# , color=(0.8, 0.898, 1)
if __name__ == '__main__':
    X = getData()
