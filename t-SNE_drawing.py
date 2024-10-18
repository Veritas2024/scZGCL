import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def read_data(file_path):
    data = pd.read_csv(file_path, header=None, delim_whitespace=True)
    return data.values


if __name__ == "__main__":
    dataset_name = "Diaphragm_Smart-seq2"  # change this if you want to see 2-D visualization of other datasets
    data = read_data(dataset_name + '_zinb_latent.txt')
    labels = read_data(dataset_name + '_zinb_label.txt')
    labels = labels.flatten()
    tsne = TSNE(n_components=2, random_state=0)
    embedding = tsne.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=30)
    plt.title('t-SNE Visualization of Cell Data')
    plt.show()
    plt.savefig(dataset_name + "Visualization.png")
