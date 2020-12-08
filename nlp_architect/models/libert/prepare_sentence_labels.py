
import absa_utils
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from tqdm import tqdm

class cluster_results:
    def __init__(self, n_clusters, centroids, labels, silhouette_score):
        self.centroids = centroids
        self.labels = labels
        self.silhouette_score = silhouette_score
        self.n_clusters = n_clusters

class DFBuilder:
    def __init__(self):        
        self.max_seq_length = 64
        self.rows = []
        self.review_id = 0
        self.review_id_map = {}
        self.domain_map = {'laptops': 'l', 'restaurants': 'r', 'device': 'd'}
        self.review_df = pd.DataFrame(columns=['review_id', 'review_text', 'domain', 'review_embedding', 'passage_id_1', 'passage_id_2', 'passage_id_3', 'passage_id_4', 'passage_id_5'])
        self.passage_df = pd.DataFrame(columns=['passage_id', 'passage_title', 'passage_text', 'passage_embedding'])
    
    
    def build_df(self, data_dir=None, domain=None):  
        model_SenBERT = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        examples = absa_utils.read_examples_from_file(data_dir, domain)

        for example in tqdm(examples):
            review = ' '.join(example.words)            
            review_embeds = model_SenBERT.encode(review)

            if review not in self.review_id_map:
                self.review_id_map[review] = self.review_id
                self.review_id += 1

                review_df_row = {'review_id': self.review_id_map[review],'review_text': review, 'review_embedding': review_embeds, 'domain': self.domain_map[domain]}
                # review_df_row.update(review_passage_ids)
                # Appending rows to df
                self.review_df = self.review_df.append(review_df_row, ignore_index=True)
                
        self.review_df.drop_duplicates('review_id', inplace=True)        

    def set_index(self):
        self.review_df = self.review_df.set_index('review_id')
        #self.passage_df = self.passage_df.set_index('passage_id')
            
    def get_dfs(self):
        return self.review_df

def main(config_yaml):

    data_root = Path(__file__).parent.absolute() / 'data' / 'csv'
    #data_dir = data_root / 'domains_all_short'
    data_dir = data_root / 'domains_all'

    df_builder = DFBuilder()  
    df_builder.build_df(data_dir=data_dir, domain='restaurants')
    df_builder.build_df(data_dir=data_dir, domain='laptops')
    df_builder.build_df(data_dir=data_dir, domain='device')
    df_builder.set_index()

    review_df = df_builder.get_dfs()
    review_df.to_pickle(data_dir/'review_df.pkl')
    
    range_n_clusters = range(2,21)
    cluster_results = cluster_data(review_df, range_n_clusters)
    
    # generate data-frame with cluster labels
    df_file_name = (data_dir/'review_with_cluster_labels_df.pkl')
    generate_df_w_cluster_labels(review_df, df_file_name,cluster_results, range_n_clusters)
    
    #plot_clusters_silhouette_scores(cluster_results)

    ################# my old code #########################
    #generate_embs_cluster_and_plot(data_dir)

def cluster_data(review_df, range_n_clusters):
    X = review_df['review_embedding'].values
    X = [np.array(element) for element in X]
    results = []

    for n_clusters in tqdm(range_n_clusters):
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        results.append(cluster_results(n_clusters, clusterer.cluster_centers_, cluster_labels, silhouette_avg))

    return results   

################# generate_df_w_cluster_labels ######################### 
def generate_df_w_cluster_labels(review_df, df_file_name, cluster_results, range_n_clusters):
    review_with_cluster_labels_df = review_df
    results_dict = {result.n_clusters: result for result in cluster_results}
    for n_clusters in tqdm(range_n_clusters):
        review_with_cluster_labels_df[f'cluster_label_{n_clusters}'] = results_dict[n_clusters].labels
        
    review_with_cluster_labels_df.to_pickle(df_file_name)
    review_with_cluster_labels_df.head()

################# plot_clusters_silhouette_scores #########################
def plot_clusters_silhouette_scores(results):
    x = []
    y = []
    for result in sorted(results, key=lambda x: x.silhouette_score, reverse=True):
        print(f"For n_clusters = {result.n_clusters} The average silhouette_score is : {result.silhouette_score}")
        x.append(result.n_clusters)
        y.append(result.silhouette_score)

    plt.bar(x, y)
    plt.xlabel("n_clusters")
    plt.ylabel("silhouette_score")
    plt.title("n_clusters vs. silhouette_score")

################# old code #########################
def generate_embs_cluster_and_plot(data_dir):
    ####### 1. Generate review embs ##############################
    model_SenBERT = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    res_sent_embs, lap_sent_embs, dev_sent_embs = generate_review_embs(data_dir, model_SenBERT) 
    
    # limit num of embeds per
    max_num_embs_per_domain = 2000
    domain_lables = np.zeros(max_num_embs_per_domain*2)
    domain_lables[max_num_embs_per_domain:2*max_num_embs_per_domain-1]=1

    all_embs = np.concatenate((res_sent_embs[0:max_num_embs_per_domain], lap_sent_embs[0:max_num_embs_per_domain]),axis=0)

    ####### 2. Clustering ##############################
    num_clusters = 10
    max_idx = clustering_embs(all_embs, num_clusters)
    
    ####### 3. Plot ##############################
    tsne_plot_clusters(all_embs, max_idx, num_clusters, domain_lables)
    
###################### generate_review_embs ##################################################  
def generate_review_embs(data_dir, model_SenBERT): 
    for dataset_name in "restaurants", "laptops", "device": 
        examples = absa_utils.read_examples_from_file(data_dir, dataset_name)
        review_texts = [' '.join(example.words) for example in examples]
        if dataset_name == "restaurants":
            res_sent_embs = model_SenBERT.encode(review_texts)
        if dataset_name == "laptops":
            lap_sent_embs = model_SenBERT.encode(review_texts)
        if dataset_name == "device":
            dev_sent_embs = model_SenBERT.encode(review_texts)    
        print(review_texts[0:3])

    return res_sent_embs, lap_sent_embs, dev_sent_embs

def clustering_embs(all_embs, num_clusters):
        # create kmeans object
        kmeans = KMeans(n_clusters=num_clusters)
        # fit kmeans object to data
        kmeans.fit(all_embs)
        # print location of clusters learned by kmeans object
        centroids = kmeans.cluster_centers_
        # project all points on centroids
        #y_km = kmeans.fit_predict(all_embs)

        #find closest points to centroids for visualizing those points as centroids
        points_as_centroid = centroids
        
        sim_vec = torch.zeros(num_clusters, all_embs.shape[0])
        i=0
        for emb in all_embs:        
            for cl_num in range(num_clusters): 
                similarity = torch.cosine_similarity(torch.from_numpy(emb).view(1,-1),torch.from_numpy(points_as_centroid[cl_num]).view(1,-1))
                sim_vec[cl_num,i] = similarity
            i=i+1

        #get indexes of closest points to centroids
        _, max_idx  = sim_vec.max(1)

        return max_idx   

def tsne_plot_clusters(all_embs, max_idx, num_clusters, domain_lables):
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_embds = tsne.fit_transform(all_embs) 
        tsne_centroids=np.zeros([num_clusters,2])
        tsne_centroids[:,:] = tsne_embds[max_idx,:]
        print("tsne centroids: ", tsne_centroids)

        tsne_df = pd.DataFrame({'X':tsne_embds[:,0], 'Y':tsne_embds[:,1], 'domain_lbl':domain_lables})
        tsne_df.head()

        #sns.scatterplot(x="X", y="Y",data=tsne_df)
        sns.scatterplot(x="X", y="Y",hue="domain_lbl",palette=['red','green'],legend='full',data=tsne_df)
        plt.scatter(tsne_centroids[:, 0], tsne_centroids[:, 1], c='black', s=50, alpha=.8)

if __name__ == "__main__":
    main('oren_debug')
   