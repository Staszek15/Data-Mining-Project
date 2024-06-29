import pandas as pd
import argparse
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path

def get_optimal_K_agg(k_max, settype):
    if settype == "train":
        df = pd.read_csv(Path('csv_files','adults_ohs_train.csv'))
    else:
        df = pd.read_csv(Path('csv_files','adults_ohs_test.csv'))
        
    print(settype)

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
    
    
    measures.to_csv(f'agnes_measures_ohs_{settype}.csv')
    
    return measures


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run Agglomerative clustering.')
    parser.add_argument('settype', help='train or test')
    args = parser.parse_args()

    get_optimal_K_agg(10, settype=args.settype)