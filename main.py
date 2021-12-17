#%%

import re # for cleaning Resume_str
import pandas as pd
import csv
import torch
import numpy as np

#%% md

### Load file

#%%

file_path = 'Resume.csv'
df = pd.read_csv(file_path, error_bad_lines=False)

#%% md

### Data observation

#%%

df.columns

#%%

df = df.drop(columns=['Resume_html'])
df.columns

#%%

df.head(5)

#%%

df.Resume_str[0]

#%%

df.info()

#%%

df_gb = df.groupby('Category')
print('Number of Category: {}'.format(df_gb.ngroups))
print(df_gb.size())

#%% md

### Preprocess data

#%%

def clean_spaces(s):
    s = ' '.join(re.split('[ ]+', s.strip()))

    return s

# Todo:
# add more preprocess function for preprocessor

def preprocessor(df):
    df['Resume_str'] = df['Resume_str'].apply(lambda x: clean_spaces(x))

    return df

#%%

df = preprocessor(df)

#%% md

### Map Resume_str to a embedding (vector)

#%%

import torch
from sentence_transformers import SentenceTransformer, util

# doc2vec
# word2vec (200, 128) -> 128, (50, 128) -> 128, (120, 128) -> 128

#%%

model = SentenceTransformer('all-distilroberta-v1')
model.max_seq_length = 512

# Corpus with resumes
Resume_corpus = df['Resume_str'].tolist()

# Calculate the embeddng for every resume_str
corpus_embeddings = model.encode(Resume_corpus)
print(corpus_embeddings.shape)

#%%

np.save('corpus_embeddings.npy', corpus_embeddings)

#%%

corpus_embeddings = np.load('corpus_embeddings.npy')

#%% md

### Apply k-Means clustering on the embeddings

#%%

import numpy as np
from sklearn.cluster import KMeans

num_clusters = df.groupby('Category').ngroups # 24
clustering_model = KMeans(n_clusters=num_clusters)

clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_ # Get the clustered label for each embedding
print(cluster_assignment.shape)

clustered_resumes = [[] for i in range(num_clusters)] # Will contain embeddings for each cluster
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_resumes[cluster_id].append(Resume_corpus[sentence_id])

#%%

print('Number of resumes in each cluster')
for i, cluster in enumerate(clustered_resumes):
    print('Cluster {}: {}'.format(i+1, len(cluster)))

#%%

# Todo:
# 1. Visualize the clustered result
# 2. Try applying other clustering methods
# 3. Other stuff that can perform on cluster, e.g. Topic modeling?

#%% md

### Visualize the clustered result

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#%%

# PCA降成2維 -> kMeans -> 畫圖
# 768 -> 2

pca = PCA(2)
corpus_embeddings_2d = pca.fit_transform(corpus_embeddings) # (2484, 768) -> (2484, 2)

num_clusters = df.groupby('Category').ngroups # 24

clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings_2d)
cluster_assignment = clustering_model.labels_

plt.figure(figsize=(15, 15))

for i in range(num_clusters):
  plt.scatter(corpus_embeddings_2d[cluster_assignment==i, 0], corpus_embeddings_2d[cluster_assignment==i, 1])

plt.legend()
plt.show()

#%%

# kMeans -> 隨機挑2維畫圖

num_clusters = df.groupby('Category').ngroups # 24

clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings) # (2484, 768)
cluster_assignment = clustering_model.labels_

plt.figure(figsize=(15, 15))

for i in range(num_clusters):
  plt.scatter(corpus_embeddings[cluster_assignment==i, 2], corpus_embeddings[cluster_assignment==i, 1])

plt.legend()
plt.show()

#%%

# PCA降成2維 -> kMeans -> 畫圖 -> 加上某職業
domain = 'CHEF'

pca = PCA(2)
corpus_embeddings_2d = pca.fit_transform(corpus_embeddings)

num_clusters = df.groupby('Category').ngroups # 24

clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings_2d)
cluster_assignment = clustering_model.labels_

plt.figure(figsize=(15, 15))

for i in range(num_clusters):
  plt.scatter(corpus_embeddings_2d[cluster_assignment==i, 0], corpus_embeddings_2d[cluster_assignment==i, 1], s=3)

df_group = df_gb.get_group('CHEF')
group_corpus_embeddings = corpus_embeddings_2d[df_group.index]
plt.scatter(group_corpus_embeddings[:, 0], group_corpus_embeddings[:, 1], label='CHEF', c='black')

df_group = df_gb.get_group('FINANCE')
group_corpus_embeddings = corpus_embeddings_2d[df_group.index]
plt.scatter(group_corpus_embeddings[:, 0], group_corpus_embeddings[:, 1], label='FINANCE', c='red')


plt.legend()
plt.show()

#%% md

### Find the cross domain (category) resumes

#%%

def get_avg_embeddings(df_gb, group_name):
    print('Group name: {}'.format(group_name))
    df_group = df_gb.get_group(group_name)

    # Resume corpus of groups
    group_corpus = df_group['Resume_str'].tolist()

    group_corpus_embeddings = model.encode(group_corpus, convert_to_tensor=True)
    print('Shape of group_corpus_embeddings: {}'.format(group_corpus_embeddings.shape))

    group_avg_embedding = torch.mean(group_corpus_embeddings, dim=0, keepdim=True)
    print('Shape of group_avg_embeddings: {}'.format(group_avg_embedding.shape))

    return group_avg_embedding

#%%

def get_avg_embeddings_revised(df_gb, corpus_embeddings, group_name):
  df_group = df_gb.get_group(group_name)
  group_corpus_embeddings = corpus_embeddings[df_group.index] # (110, 768)
  group_avg_embedding = np.mean(group_corpus_embeddings, axis=0)
  return group_avg_embedding

#%%

def search_resumeID(query_embedding, corpus_embeddings, top_k):
  hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
  hits = hits[0]
  for hit in hits:
    print(df.ID[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

#%%

corpus_embeddings = model.encode(Resume_corpus)

A  = [ 1, 2, 3, ...] # 768
B  = [ 8, 2, 1, ...] # 768
avg = [4.5, 2, 2, ...]

#%%

ACCNT_avg_embedding = get_avg_embeddings_revised(df_gb, corpus_embeddings, 'ACCOUNTANT')
IT_avg_embedding = get_avg_embeddings_revised(df_gb, corpus_embeddings, 'INFORMATION-TECHNOLOGY')

#%%

query_embedding = (ACCNT_avg_embedding + IT_avg_embedding) / 2
search_resumeID(query_embedding, corpus_embeddings, top_k=5)

#%%

# Todo:
# 1. Cross 3 or more domain (categories)
# 2. Find qualitative example (we can show in report/presentation)
# 3. Other vector (embedding) operation to perform, i.e., other task

#%% md

### Cross 3 or more domain (categories)

#%%

def cross_domain(df_gb, domains, corpus_embeddings, top_k):
  avg_embeddings = []
  for domain in domains:
    avg_embeddings.append(get_avg_embeddings_revised(df_gb, corpus_embeddings, domain))
  query_embedding = np.mean(avg_embeddings, axis=0) # 768
  search_resumeID(query_embedding, corpus_embeddings, top_k)

#%%

cross_domain(df_gb, ['ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE'], corpus_embeddings, 5)
