import pandas as pd
from pathlib import Path
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

def reduce_by_MDS(train, test, max_dims, settype):
    print(settype, " start")
    distance_matrix = pdist(train, metric='euclidean')
    distance_matrix_square = squareform(distance_matrix)
    print("policzone macierze")
    stresses = []
    for dim in range(1, max_dims + 1):
        print("dim=",dim)
        mds = MDS(n_components=dim, dissimilarity="precomputed", random_state=42)
        embedding = mds.fit_transform(distance_matrix_square)
        stresses.append(mds.stress_)

    results = pd.DataFrame({'stress':stresses})
    results.index += 1
    results.to_csv(f'MDS_stress{settype}_train.csv')
    embedding.to_csv(f'MDS_reduced{settype}_train.csv')
    
    
if __name__ == '__main__':
    
    train_ohs = pd.read_csv(Path('..','csv_files','adults_ohs_train.csv'))
    test_ohs = pd.read_csv(Path('..','csv_files','adults_ohs_test.csv'))

    train_ohn = pd.read_csv(Path('..','csv_files','adults_ohn_train.csv'))
    test_ohn = pd.read_csv(Path('..','csv_files','adults_ohn_test.csv'))
    
    reduce_by_MDS(train_ohs, test_ohs, 40, 'ohs')
    reduce_by_MDS(train_ohn, test_ohn, 40, 'ohn')