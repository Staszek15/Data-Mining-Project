import pandas as pd
from pathlib import Path
import cupy as cp   
import argparse 

from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def K_Prot(k_max, settype):
    
    if settype == "train":
        df = pd.read_csv(Path('csv_files','adults_norm_train.csv'))
    else:
        df = pd.read_csv(Path('csv_files','adults_norm_test.csv'))
        
    print(settype)
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(exclude="number").columns
    print(categorical_cols)
    categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]
    
    wcss = []
    
    kprototype = KPrototypes(n_jobs=4, n_clusters=1, random_state=42)
    clusters = kprototype.fit_predict(df.to_numpy(), categorical=categorical_indices)
    print("za peirwszym klastrowaniem")
    wcss.append(kprototype.cost_)
    print("Za elbow")

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
    res_df.to_csv(f'k-prototype_norm_{settype}.csv')
    
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run K-Prototype clustering.')
    parser.add_argument('settype', help='train or test')
    args = parser.parse_args()

    K_Prot(10, settype=args.settype)