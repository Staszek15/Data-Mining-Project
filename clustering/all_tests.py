import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn_extra.cluster import KMedoids
from kmodes.kprototypes import KPrototypes


def optimal_K_means(train, test, opt_k, settype):
    
    kmeans = KMeans(n_clusters=opt_k, max_iter=5000, random_state=42)
    kmeans.fit_predict(train)  
    
    train_clusters = kmeans.predict(train)
    test_clusters = kmeans.predict(test)
    train['cluster'] = train_clusters
    test['cluster'] = test_clusters
    
    silhouette_scores = silhouette_score(test, test_clusters)
    db_scores = davies_bouldin_score(test, test_clusters)
    ch_scores = calinski_harabasz_score(test, test_clusters)
    wcss = kmeans.inertia_
    
    measures = pd.DataFrame({'elbow':[wcss],
                            'silh':[silhouette_scores],
                            'dbi':[db_scores],
                            'ch':[ch_scores]})

    measures.index += 1
    
    measures.to_csv(f'kmeans_{opt_k}_{settype}.csv')
    train.to_csv(f'kmeans_{opt_k}_{settype}_labelled_train.csv')
    test.to_csv(f'kmeans_{opt_k}_{settype}_labelled_test.csv')
    
    
    
def optimal_K_medoids(train, test, opt_k, settype):
    
    kmedoids = KMedoids(n_clusters=opt_k, max_iter=5000, random_state=42,
                        init="random")
    kmedoids.fit_predict(train)  
    
    train_clusters = kmedoids.predict(train)
    test_clusters = kmedoids.predict(test)
    train['cluster'] = train_clusters
    test['cluster'] = test_clusters
    
    silhouette_scores = silhouette_score(test, test_clusters)
    db_scores = davies_bouldin_score(test, test_clusters)
    ch_scores = calinski_harabasz_score(test, test_clusters)
    wcss = kmedoids.inertia_
    
    measures = pd.DataFrame({'elbow':[wcss],
                            'silh':[silhouette_scores],
                            'dbi':[db_scores],
                            'ch':[ch_scores]})

    measures.index += 1
    
    measures.to_csv(f'kmedoids_{opt_k}_{settype}.csv')
    train.to_csv(f'kmedoids_{opt_k}_{settype}_labelled_train.csv')
    test.to_csv(f'kmedoids_{opt_k}_{settype}_labelled_test.csv')

def optimal_K_prototype(train, test, opt_k, settype):
    
    categorical_cols_train = train.select_dtypes(exclude="number").columns
    categorical_indices_train = [train.columns.get_loc(col) for col in categorical_cols_train]
    
    kprototype = KPrototypes(n_clusters=opt_k, max_iter=5000, random_state=42)
    train_clusters = kprototype.fit_predict(train, categorical=categorical_indices_train)  # Dopasowanie modelu i przypisanie klastrów do danych treningowych
    
    categorical_cols_test = test.select_dtypes(exclude="number").columns
    categorical_indices_test = [test.columns.get_loc(col) for col in categorical_cols_test]
    
    test_clusters = kprototype.predict(test, categorical=categorical_indices_test)  # Przypisanie klastrów do danych testowych
    
    wcss = kprototype.cost_
    
    train['cluster'] = train_clusters
    test['cluster'] = test_clusters
    
    measures = pd.DataFrame({'elbow':[wcss]})
    measures.index += 1
    
    measures.to_csv(f'kprototype_{opt_k}_{settype}.csv')
    train.to_csv(f'kprototype_{opt_k}_{settype}_labelled_train.csv')
    test.to_csv(f'kprototype_{opt_k}_{settype}_labelled_test.csv')
    
    return measures

def optimal_agglomerative(train, test, opt_k, settype):
    
    agglo = AgglomerativeClustering(n_clusters=opt_k)
    
    train_clusters = agglo.fit_predict(train)
    #test_clusters = agglo.predict(test)
    train['cluster'] = train_clusters
    #test['cluster'] = test_clusters
    
    """silhouette_avg = silhouette_score(test, test_clusters)
    db_score = davies_bouldin_score(test, test_clusters)
    ch_score = calinski_harabasz_score(test, test_clusters)
    
    measures = pd.DataFrame({'silh':[silhouette_avg],
                             'dbi':[db_score],
                             'ch':[ch_score]})
    
    measures.index += 1
    
    measures.to_csv(f'agglomerative_{opt_k}_{settype}.csv')"""
    train.to_csv(f'agglomerative_{opt_k}_{settype}_labelled_train.csv')
    #test.to_csv(f'agglomeraive_{opt_k}_{settype}_labelled_test.csv')
    #return measures



if __name__ == '__main__':
    
    norm_train = pd.read_csv(Path('csv_files','adults_norm_train.csv'))
    norm_test = pd.read_csv(Path('csv_files','adults_norm_test.csv'))

    std_train = pd.read_csv(Path('csv_files','adults_std_train.csv'))
    std_test = pd.read_csv(Path('csv_files','adults_std_test.csv'))

    ohn_train = pd.read_csv(Path('csv_files','adults_ohn_train.csv'))
    ohn_test = pd.read_csv(Path('csv_files','adults_ohn_test.csv'))

    ohs_train = pd.read_csv(Path('csv_files','adults_ohs_train.csv'))
    ohs_test = pd.read_csv(Path('csv_files','adults_ohs_test.csv'))
    
    #optimal_K_means(ohs_train, ohs_test, 8, 'ohs')
    #optimal_K_means(ohn_train, ohn_test, 3, 'ohn')
    
    #optimal_K_medoids(ohs_train, ohs_test, 3, 'ohs')
    #optimal_K_medoids(ohn_train, ohn_test, 5, 'ohn')
    
    #optimal_K_prototype(std_train, std_test, 6, 'std')
    #optimal_K_prototype(norm_train, norm_test, 2, 'norm')
    
    optimal_agglomerative(ohs_train, ohs_test, 8, 'ohs')
    #optimal_agglomerative(ohn_train, ohn_test, 2, 'ohn')
