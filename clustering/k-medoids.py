import pandas as pd
from pathlib import Path
import numpy as np
import argparse

from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn_extra.cluster import KMedoids


def get_optimal_K_med(k_max, settype):
    if settype == "train":
        df = pd.read_csv(Path('csv_files','adults_ohn_train.csv'))
    else:
        df = pd.read_csv(Path('csv_files','adults_ohn_test.csv'))

    print(settype)
    wcss = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []
    #ar_scores = []
    print(1)
    # for k = 1
    kmedoids = KMedoids(n_clusters=1, max_iter=5000, random_state=42)
    kmedoids.fit_predict(df)
    print("za peirwszym klastrowaniem")
    wcss.append(kmedoids.inertia_)
    print("Za elbow")
    silhouette_scores.append(None)
    print("za sil")
    db_scores.append(None)
    print("xa db")
    ch_scores.append(None)
    print("za ch")

    for k in range(2, k_max+1):
        print("k=",str(k))
        kmedoids = KMedoids(n_clusters=k, max_iter=5000, random_state=42, init='random')
        labels = kmedoids.fit_predict(df)
        print("labelski done")
        wcss.append(kmedoids.inertia_)
        print('za elbow')
        silhouette_scores.append(silhouette_score(df, labels))
        print('za silh')
        db_scores.append(davies_bouldin_score(df, labels))
        ch_scores.append(calinski_harabasz_score(df, labels))
        #ar_scores.append(adjusted_rand_score(df, labels))


    # Prepare data for visualization:
    measures = pd.DataFrame({'elbow':wcss,
                         'silh':silhouette_scores,
                         'dbi':db_scores,
                         'ch':ch_scores})
                         #'ar':ar_scores})
    measures.index += 1


    measures.to_csv(f'K-medoids_measures_ohn_{settype}.csv')

    return measures


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run K-Medoids clustering.')
    parser.add_argument('settype', help='train or test')
    args = parser.parse_args()

    get_optimal_K_med(10, settype=args.settype)