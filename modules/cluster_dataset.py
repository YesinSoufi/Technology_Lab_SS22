# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import librosa
import IPython.display

plt.rcParams['figure.figsize'] = (20, 20)


# %%
# expect dataframe with features
def cluster_data(df_unlabeled_dataset, cluster: int):
    df_dataset = df_unlabeled_dataset
    featureArray = create_featureArray(df_dataset)
    norm_featureArray = norm_values(featureArray)
    df_dataset['label'] = kmean_cluster(norm_featureArray, cluster)

    # plot
    # plot_features(feature1=4, feature2=5, norm_features=norm_featureArray)
    plot_cluster(norm_features=norm_featureArray, labels=df_dataset['label'])

    return df_dataset


# %%
# create 2D numpy.array with all features from all samples
# only feature data, no audio data
def create_featureArray(df_dataset_unlabeled):
    # featureArray = np.array(df_dataset_unlabeled.iloc[:, 3:9])
    featureArray = np.array(df_dataset_unlabeled.iloc[:, 4:6])
    # only for sample_features_df.csv
    # featureArray = np.array(df_dataset_unlabeled.iloc[:, 4:10])
    print('featureArray: ' + str(featureArray))
    return featureArray


# %%
# normalize all features, ready to cluster
def norm_values(featureArray):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # min_max_scaler.fit(featureArray)
    norm_features = min_max_scaler.fit_transform(featureArray)
    # norm_features = min_max_scaler.fit(featureArray)
    # norm_features = min_max_scaler.transform(featureArray)

    print('norm_features: ' + str(norm_features))
    # print(features_scaled.min(axis=0))
    # print(features_scaled.max(axis=0))
    return norm_features


# %%
# hilfsfunktion, plot von zwei feature werten
def plot_features(feature1: int, feature2: int, norm_features):
    plt.scatter(norm_features[:, feature1], norm_features[:, feature2])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')


# %%
# choose clustering algorithm
# try with kmean-clustering
# choose how many cluster
def kmean_cluster(norm_features, cluster: int):
    model = sklearn.cluster.KMeans(n_clusters=cluster)
    labels = model.fit_predict(norm_features)
    # print(labels)
    # print(type(features_scaled))
    # print(features_scaled)
    return labels


# %%
# sample for affinity propagation
def affinityPropagation_cluster(norm_features):
    model = sklearn.cluster.AffinityPropagation()
    labels = model.fit_predict(norm_features)
    # print(labels)
    return labels


# %%
# plot labeled data
# choose two features to validate
# add or remove lines to increase/ decrease classes
def plot_cluster(norm_features, labels):
    feature1 = 0
    feature2 = 1

    plt.scatter(norm_features[labels == 0, feature1], norm_features[labels == 0, feature2], c='b')
    plt.scatter(norm_features[labels == 1, feature1], norm_features[labels == 1, feature2], c='r')
    plt.scatter(norm_features[labels == 2, feature1], norm_features[labels == 2, feature2], c='y')
    # plt.scatter(norm_features[labels==3,feature1], norm_features[labels==3,feature2], c='g')
    # plt.scatter(norm_features[labels==4,feature1], norm_features[labels==4,feature2], c='c')
    # plt.scatter(norm_features[labels==5,feature1], norm_features[labels==5,feature2], c='b')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(('Class 0', 'Class 1', 'Class 2'))
