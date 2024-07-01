import pandas as pd
from pathlib import Path
import numpy as np
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn_extra.cluster import KMedoids
from kmodes.kprototypes import KPrototypes


def K_Medoids(df, k_max, stdtype):

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
    measures.to_csv(f'PCA-medoids_measures_{stdtype}_train.csv')

    return measures


def K_Means(df, k_max, stdtype):

    wcss = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []
    
    # for k = 1
    kmeans = KMeans(n_clusters=1, max_iter=5000, random_state=42)
    kmeans.fit_predict(df)        
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(None)
    db_scores.append(None)
    ch_scores.append(None)
    
    for k in range(2, k_max+1):
        print("k=",str(k))
        kmeans = KMeans(n_clusters=k, max_iter=5000, random_state=42)
        labels = kmeans.fit_predict(df)        
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))
        db_scores.append(davies_bouldin_score(df, labels))
        ch_scores.append(calinski_harabasz_score(df, labels))

    # Prepare data for visualization:
    measures = pd.DataFrame({'elbow':wcss,
                         'silh':silhouette_scores,
                         'dbi':db_scores,
                         'ch':ch_scores})
    measures.index += 1   
    measures.to_csv(f'PCA-means_measures_{stdtype}.csv')
    
    return measures

    
def K_Prot(df, k_max, stdtype):
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(exclude="number").columns
    print(categorical_cols)
    categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]
    
    wcss = []
    
    kprototype = KPrototypes(n_jobs=4, n_clusters=1, random_state=42)
    clusters = kprototype.fit_predict(df.to_numpy(), categorical=categorical_indices)
    wcss.append(kprototype.cost_)

    for k in range(2, k_max + 1):
        print(f"Clustering for k = {k}")
        kprototype = KPrototypes(n_jobs=4, n_clusters=k, random_state=42)
        clusters = kprototype.fit_predict(df.to_numpy(), categorical=categorical_indices)
        
        # Append cost (WCSS)
        wcss.append(kprototype.cost_)
        
    # Prepare results in a DataFrame
    res_df = pd.DataFrame({
        'elbow': wcss
        })
    res_df.index += 1
    
    # Save results to CSV
    res_df.to_csv(f'PCA-prototype_measures_{stdtype}.csv')
    
    return res_df


def agg(df, k_max, stdtype):

    silhouette_scores = []
    db_scores = []
    ch_scores = []
    #ar_scores = []
    print(1)
    # for k = 1
    agg = AgglomerativeClustering(n_clusters=1)
    agg.fit_predict(df)  
    print("za peirwszym klastrowaniem")      
   
    silhouette_scores.append(None)
    print("za sil")
    db_scores.append(None)
    print("xa db")
    ch_scores.append(None)
    print("za ch")
    
    for k in range(2, k_max+1):
        print("k=",str(k))
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(df)        
        print("labelski done")

        silhouette_scores.append(silhouette_score(df, labels))
        print('za silh')
        db_scores.append(davies_bouldin_score(df, labels))
        ch_scores.append(calinski_harabasz_score(df, labels))
        #ar_scores.append(adjusted_rand_score(df, labels))
        
    # Prepare data for visualization:
    measures = pd.DataFrame({'silh':silhouette_scores,
                            'dbi':db_scores,
                            'ch':ch_scores})
                         #'ar':ar_scores})
    measures.index += 1
    measures.to_csv(f'PCA-agg_measures_{stdtype}.csv')
    
    return measures


if __name__ == "__main__":
    
    train_ohs = pd.read_csv(Path('csv_files','PCA2_ohs_train.csv'), index_col=0)
    test_ohs = pd.read_csv(Path('csv_files','PCA2_ohs_test.csv'), index_col=0)
    
    train_ohn = pd.read_csv(Path('csv_files','PCA2_ohn_train.csv'), index_col=0)
    test_ohn = pd.read_csv(Path('csv_files','PCA2_ohn_test.csv'), index_col=0)
    
    print("###### medoids ###### ")
    K_Medoids(train_ohs, 10, 'ohs')
    K_Medoids(train_ohn, 10, 'ohn')
    
    print("###### means ###### ")
    K_Means(train_ohs, 10, 'ohs')
    K_Means(train_ohn, 10, 'ohn')
    
    print("###### prot ###### ")
    #K_Prot(train_std, 10, 'std')
    #K_Prot(train_norm, 10, 'norm')
    
    print("###### agg ###### ")
    agg(train_ohs, 10, 'ohs')
    agg(train_ohn, 10, 'ohn')
    
    

    