# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   date：          6/1/2020
-------------------------------------------------
   Change Activity:
                   6/1/2020:
-------------------------------------------------
"""
from getData import getData
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


def cluster():
    try:
        X = np.load("achieve/ProcessedData.npy")
    except:
        X = getData()
    # standard the data range from 0 to 1
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # pca to judge the n component in the GMM
    # first print the histogram of data
    plt.subplot(211)
    clf = PCA(n_components=X.shape[1])
    clf.fit(X)
    plt.bar(range(clf.n_components_), clf.explained_variance_ratio_)
    plt.xlabel("PCA features")
    plt.ylabel("variables %")
    plt.xticks(range(0, X.shape[1]))
    # // scatter the data to get the number of conpnnent
    plt.subplot(212)
    choice = np.random.choice(X.shape[0], int(X.shape[0] / 2800))
    pca_X = clf.fit_transform(X)
    plt.scatter(pca_X[choice, 0], pca_X[choice, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("image/PCA.svg", dpi=1200, format='svg')
    plt.show()
    del choice, clf, ss

    # Scree  plot showing a slow decrease
    ks = range(1, 10)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        clf = GaussianMixture(n_components=k)

        # Fit model to samples
        clf.fit(pca_X[:, :3])

        # Append the inertia to the list of inertias
        inertias.append(clf.lower_bound_)
    plt.plot(ks, inertias,'-o',)
    plt.xlabel('Number of Clusters, k')
    plt.ylabel('Lower Bound Value on the Log-Likelihood')
    plt.xticks(ks)
    plt.savefig("image/GMM_low_bound.svg", dpi=1200, format='svg')
    plt.show()
    # Kmeans to cluster the data
    clf = GaussianMixture(n_components=3)
    clf.fit(X)
    label = clf.predict(X)
    # set the properties of scatter3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Name = ["condition1", "condition2", "condition3"]
    for i in range(3):
        ax.scatter(X[label == i, 0], X[label == i, 1], X[label == i, 2], label=Name[i], rasterized=True)
    ax.set_xlabel("Wind Speed m/s")
    ax.set_ylabel("Grid Power Kw/h")
    ax.set_zlabel("Rotor Speed m/s")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("image/gmm.svg", dpi=1200, format='svg')
    plt.show()

    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))
    # pca algorithm to reduce dim
    clf = PCA(n_components=3)
    X = clf.fit_transform(X)
    # 3 cluster multiple X
    percent_label = [np.sum(label == 0), np.sum(label == 1), np.sum(label == 2)]
    percent_label = np.flip(np.sort(percent_label))
    X = (percent_label[0] / label.shape[0]) * X[:, 0] + (percent_label[1] / label.shape[0]) * X[:, 1] + (
            percent_label[2] / label.shape[0]) * X[:, 2]
    # set the high border
    # X[X > 0.2] = 0.2
    # plt.plot(X)
    # plt.show()
    time_slide_list = []
    Finally_label = np.zeros_like(label)
    for i in range(15):
        time_slide_list.append(np.mean(X[i:((i + 1) * 8000)]))

    for i in range(15):
        if time_slide_list[i] <= np.mean(X):
            Finally_label[i:((i + 1) * 10000)] = 0
        elif time_slide_list[i] <= np.mean(X) + np.var(X):
            Finally_label[i:((i + 1) * 10000)] = 1
        else:
            Finally_label[i:((i + 1) * 10000)] = 2

    plt.plot(time_slide_list, label="Similarity Measure")
    plt.plot([0, 14], [np.mean(X), np.mean(X)], label="Mean")
    plt.plot([0, 14], [np.mean(X) + np.var(X), np.mean(X) + np.var(X)], label="Mean+Variance")
    plt.legend(loc="upper right")
    plt.xlim((0, 14))
    plt.xlabel("Time")
    plt.ylabel("Reliability Assessment")
    plt.tight_layout()
    plt.savefig("image/HealthEstimate.svg", dpi=1200, format='svg')
    plt.show()

    np.save("achieve/label.npy", Finally_label)


if __name__ == '__main__':
    cluster()
