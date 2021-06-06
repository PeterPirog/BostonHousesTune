import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

from sklearn.cluster import DBSCAN,KMeans,AffinityPropagation,AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

if __name__ == '__main__':
    verbose=False
    # make all dataframe columns visible
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('data/XY_train_enc.csv')
    columns = df.columns
    #df=StandardScaler().fit_transform(df)

    #target scaling
    ct = ColumnTransformer([("scaler", StandardScaler(), ['SalePrice'])])

    df_original_price=df['SalePrice'].copy()
    df['SalePrice']=ct.fit_transform(df)



    #df=pd.DataFrame(df,columns=columns)
    #print(df.head())

    # Compute DBSCAN
    #clustering = KMeans().fit(df)
    #clustering = DBSCAN(eps=0.3, min_samples=100).fit(df)
    #clustering=AffinityPropagation(random_state=5).fit(df)
    clustering=AgglomerativeClustering(n_clusters=10,compute_distances=True).fit(df)


    #get class labels
    labels=clustering.labels_

    #return to df type and concatenate
    df=pd.DataFrame(df,columns=columns,dtype=np.float32) #back df from np array to df
    #back 'SalePrice' values to original
    df['SalePrice']=df_original_price
    out=pd.DataFrame(np.array(labels,dtype=np.int32),columns=['cluster'])
    df=pd.concat([df,out],axis=1)





    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(clustering, truncate_mode='level', p=4)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    #save results
    df.to_csv('data/XY_enc_cluster.csv', index=False)
    df.to_excel('data/XY_enc_cluster.xlsx',sheet_name='X_encoded',index=False)
    

