import torch
import scanpy as sc
import numpy as np
from torch_geometric.data import HeteroData,Data
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from scipy.stats import pearsonr

def read_data(name):

    path = f'./data/{name}.h5'
    adata = sc.read(path)

    return adata

def normalize(adata, HVG=0.2, filter=True, size_factors=True, logtrans_input=True, normalize_input=True):

    if filter:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata,qc_vars=['mt'],percent_top=None,log1p=False,inplace=True)
        adata = adata[adata.obs.pct_counts_mt < 5]
    n = int(adata.X.shape[1] * HVG)
    hvg_gene_idx = np.argsort(adata.X.var(axis=0))[-n:]
    adata = adata[:,hvg_gene_idx]


    adata.raw = adata.copy()

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

def knn_to_edge_index(inds):
    num = inds.shape[0]
    k = inds.shape[1]
    edge_index = torch.zeros(size=(2,num*k),dtype=torch.int64)
    for i,ind in enumerate(inds):
        for j,n_id in enumerate(ind):
            edge_index[0][i*k + j] = i
            edge_index[1][i*k + j] = n_id
    return edge_index


def construct_graph(features,k=60,method='p'):
    
    if method == 'euclidean':
        sim = euclidean_distances(features)
        sim = 1 / (sim + 1)

    elif method == 'cos':
        sim = cosine_similarity(features)
    
    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        sim = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    num = features.shape[0]
    inds = []
    for i in range(sim.shape[0]):
        ind = np.argpartition(sim[i, :], -k)[-k:]
        inds.append(ind)

    inds = np.array(inds)
    edge_index = knn_to_edge_index(inds)

    graph = Data(x = torch.tensor(features,dtype=torch.float32),edge_index=edge_index)
    graph.n_id = torch.arange(num)

    return graph

def error_rate(features,group,k=60,method='p'):

    if method == 'euclidean':
        sim = euclidean_distances(features)
        sim = 1 / (sim + 1)

    elif method == 'cos':
        sim = cosine_similarity(features)
    
    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        sim = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    num = features.shape[0]
    inds = []
    for i in range(sim.shape[0]):
        ind = np.argpartition(sim[i, :], -k)[-k:]
        inds.append(ind)
    inds = np.array(inds)
    count = 0
    for i,v in enumerate(inds):
        for vv in v:
            if group.iloc[i] != group.iloc[vv]:
                count += 1
    return count / (num * k)