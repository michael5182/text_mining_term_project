# %%

import re  # for cleaning Resume_str
from collections import Counter
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
import torch
from sentence_transformers import SentenceTransformer, util


# %% md

### Load file

# %%

file_path = 'Resume.csv'
df = pd.read_csv(file_path, error_bad_lines=False)

# %% md

### Data observation

# %% load dataframe.

df = df.drop(columns=['Resume_html'])
print(
    f"The columns of dataframe: {df.columns}")
print(
    f"The first rows of dataframe: {df.head(5)}"
)
print(
    df.Resume_str[0]
)
print(
    df.info()
)
# %% labeling.
d_label: dict = {s: i for i, s in enumerate(set(df['Category'].tolist()))}
df['Category_label'] = df['Category'].map(d_label).fillna(np.nan)
assert sorted(Counter(df['Category']).values()) == sorted(Counter(df['Category_label']).values())

# print(
#     json.dumps(
#         Counter(df['Category']), indent=4)
# )
# print(
#     json.dumps(
#         Counter(df['Category_label']), indent=4)
# )
# %%

df_gb = df.groupby('Category')
print('Number of Category: {}'.format(df_gb.ngroups))
print(df_gb.size())
# %% ploting of counter of application.
df_gb.size().plot(kind='bar', title='Counter of Application')
plt.show()

# %%
# Corpus with resumes
Resume_corpus = df['Resume_str'].tolist()



# %% md

### Preprocess data

# %%


def clean_spaces(s):
    s = ' '.join(re.split('[ ]+', s.strip()))

    return s


# Todo:
# add more preprocess function for preprocessor


def preprocessor(df):
    df['Resume_str'] = df['Resume_str'].apply(lambda x: clean_spaces(x))

    return df


df = preprocessor(df)


# %% md

### Map Resume_str to a embedding (vector)

# %%

# doc2vec
# word2vec (200, 128) -> 128, (50, 128) -> 128, (120, 128) -> 128

# %% use the pre-trained model to get the word embeddings.

model = SentenceTransformer('all-distilroberta-v1')
model.max_seq_length = 512

# Calculate the embeddng for every resume_str
corpus_embeddings = model.encode(Resume_corpus)
print(corpus_embeddings.shape)

np.save('corpus_embeddings.npy', corpus_embeddings)

# %% load the word embeddings.

corpus_embeddings = np.load('corpus_embeddings.npy')

# %% md

### Apply k-Means clustering on the embeddings

# %% use the elbow method to choose the optimal cluster number.
pair_clusters = 25
l_compared: list = list()
for k in range(2, pair_clusters):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(corpus_embeddings)
    l_compared.append(kmeans.inertia_)

l_inertia: list = [l_compared[s] - l_compared[s + 1] for s in range(len(l_compared) - 1)]
num_clusters_optimal = int(np.argmin(l_inertia))
print(f"optimal num_clesters: {num_clusters_optimal}")
num_clusters: int = num_clusters_optimal
print(f"--- setting the num_clusters: {num_clusters} ---")

fig, ax = plt.subplots()
ax.plot(range(2, pair_clusters), l_compared)
# ax.set_title("elbow function for selecting cluster number parameter.")
ax.set_title(f"optimal clustering number: {num_clusters_optimal}")
plt.show()

# %% set the clusters number manually.
num_clusters: int = 24
print(f"--- setting the num_clusters: {num_clusters} ---")

# %% clustering.

# num_clusters = df.groupby('Category').ngroups  # 24
clustering_model = KMeans(n_clusters=num_clusters)

clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_  # Get the clustered label for each embedding
print(cluster_assignment.shape)

clustered_resumes = [[] for i in range(num_clusters)]  # Will contain embeddings for each cluster
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_resumes[cluster_id].append(Resume_corpus[sentence_id])

print('Number of resumes in each cluster')
for i, cluster in enumerate(clustered_resumes):
    print('Cluster {}: {}'.format(i + 1, len(cluster)))

print(
    f"Adjusted Rand Index: {metrics.adjusted_rand_score(df['Category_label'], cluster_assignment)}"
)

# %%

# Todo:
# 1. Visualize the clustered result
# 2. Try applying other clustering methods
# 3. Other stuff that can perform on cluster, e.g. Topic modeling?

# %% md

### Visualize the clustered result

# %% use t-SNE to demostrate the result of clustering.

corpus_embeddings_2d_tsne = TSNE(
    n_components=2,
    learning_rate='auto',
    init='random').fit_transform(corpus_embeddings)  # (2484, 768) -> (2484, 2)

mpl.rcParams['lines.markersize'] = 2
fig, ax = plt.subplots(layout='constrained')
ax.set_title(f"The plotting of t-SNE method with {num_clusters} clusters.")
for i in range(num_clusters):
    plt.scatter(
        corpus_embeddings_2d_tsne[cluster_assignment == i, 0],
        corpus_embeddings_2d_tsne[cluster_assignment == i, 1]
    )
plt.show()

# %% PCA降成2維 -> kMeans -> 畫圖
# 768 -> 2

pca = PCA(2)
corpus_embeddings_2d = pca.fit_transform(corpus_embeddings)  # (2484, 768) -> (2484, 2)

clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings_2d)
cluster_assignment = clustering_model.labels_

# plt.figure(figsize=(15, 15))
mpl.rcParams['lines.markersize'] = 2
for i in range(num_clusters):
    plt.scatter(
        corpus_embeddings_2d[cluster_assignment == i, 0],
        corpus_embeddings_2d[cluster_assignment == i, 1]
    )

plt.title(f"The plotting of PCA method with {num_clusters} clusters.")
plt.show()

# %% kMeans -> 隨機挑2維畫圖  * 隨機挑兩維顯示的方法，在多維空間下可能無意義，因此先註解掉。

# clustering_model = KMeans(n_clusters=num_clusters)
# clustering_model.fit(corpus_embeddings)  # (2484, 768)
# cluster_assignment = clustering_model.labels_
#
# plt.figure(figsize=(15, 15))
#
# for i in range(num_clusters):
#   plt.scatter(
#       corpus_embeddings[cluster_assignment == i, 2],
#       corpus_embeddings[cluster_assignment == i, 1]
#   )
#
# plt.legend()
# plt.show()

# %% PCA降成2維 -> kMeans -> 畫圖 -> 加上某職業
domain1 = 'CHEF'
domain2 = 'FINANCE'

pca = PCA(2)
corpus_embeddings_2d = pca.fit_transform(corpus_embeddings)

# num_clusters = df.groupby('Category').ngroups  # 24

clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings_2d)
cluster_assignment = clustering_model.labels_

mpl.rcParams['lines.markersize'] = 2
# plt.figure(figsize=(15, 15))

for i in range(num_clusters):
    plt.scatter(
        corpus_embeddings_2d[cluster_assignment == i, 0],
        corpus_embeddings_2d[cluster_assignment == i, 1], s=3, c='grey'
    )

df_group = df_gb.get_group(domain1)
group_corpus_embeddings = corpus_embeddings_2d[df_group.index]
plt.scatter(group_corpus_embeddings[:, 0], group_corpus_embeddings[:, 1], label='CHEF', c='black', s=20)

df_group = df_gb.get_group(domain2)
group_corpus_embeddings = corpus_embeddings_2d[df_group.index]
plt.scatter(group_corpus_embeddings[:, 0], group_corpus_embeddings[:, 1], label='FINANCE', c='red', s=20)

plt.title(f"The plotting of clustering for `{domain1}` and `{domain2}` features")
plt.legend()
plt.show()


# %% md

### Find the cross domain (category) resumes

# %%

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


# %%

def get_avg_embeddings_revised(df_gb, corpus_embeddings, group_name):
    df_group = df_gb.get_group(group_name)
    group_corpus_embeddings = corpus_embeddings[df_group.index]  # (110, 768)
    group_avg_embedding = np.mean(group_corpus_embeddings, axis=0)
    return group_avg_embedding


# %%

def search_resumeID(query_embedding, corpus_embeddings, top_k):
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]
    for hit in hits:
        print(
            df.Category[hit['corpus_id']],
            df.ID[hit['corpus_id']],
            "(Score: {:.4f})".format(hit['score'])
        )


# %%

# corpus_embeddings = model.encode(Resume_corpus)

A = [1, 2, 3, ...]  # 768
B = [8, 2, 1, ...]  # 768
avg = [4.5, 2, 2, ...]

# %%

ACCNT_avg_embedding = get_avg_embeddings_revised(df_gb, corpus_embeddings, 'ACCOUNTANT')
IT_avg_embedding = get_avg_embeddings_revised(df_gb, corpus_embeddings, 'INFORMATION-TECHNOLOGY')

# %%

query_embedding = (ACCNT_avg_embedding + IT_avg_embedding) / 2
search_resumeID(query_embedding, corpus_embeddings, top_k=5)

# %%

# Todo:
# 1. Cross 3 or more domain (categories)
# 2. Find qualitative example (we can show in report/presentation)
# 3. Other vector (embedding) operation to perform, i.e., other task

# %% md

### Cross 3 or more domain (categories)

# %%

def cross_domain(df_gb, domains, corpus_embeddings, top_k):
    avg_embeddings = []
    for domain in domains:
        avg_embeddings.append(get_avg_embeddings_revised(df_gb, corpus_embeddings, domain))
    query_embedding = np.mean(avg_embeddings, axis=0)  # 768
    search_resumeID(query_embedding, corpus_embeddings, top_k)


# %%

cross_domain(df_gb, ['ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE'], corpus_embeddings, 5)

# %% completing of finding cross-domain resume.

l_result: list = list()
for domain in df_gb.size().keys():

    embeddings_group = corpus_embeddings[df_gb.get_group(domain).index]
    embeddings_mean = np.mean(embeddings_group, axis=0)

    # find d_hat.
    result_hat = [
        (i,
         np.dot(embeddings_mean, embeddings_group[i]) / (np.linalg.norm(embeddings_mean) * np.linalg.norm(embeddings_group[i]))
        )
        for i in range(embeddings_group.shape[0])
    ]

    d_hat_id_raw, result_cosine_hat = max(result_hat, key=lambda a: a[1])
    print(f"result of d_hat:", d_hat_id_raw, result_cosine_hat)
    d_hat = embeddings_group[d_hat_id_raw]
    d_hat_id = df_gb.get_group(domain).iloc[d_hat_id_raw].ID

    # find d_star.
    result_star = [
        (i,
         np.dot(d_hat, embeddings_group[i]) / (np.linalg.norm(d_hat) * np.linalg.norm(embeddings_group[i]))
         )
        for i in range(embeddings_group.shape[0])
    ]

    d_star_id_raw, result_cosine_star = min(result_star, key=lambda a: a[1])
    print("result of d_star:", d_star_id_raw, result_cosine_star)
    d_star = embeddings_group[d_star_id_raw]
    d_star_id = df_gb.get_group(domain).iloc[d_star_id_raw].ID

    # %% find the closed clustering of d_star.
    result_each_category_embedding_mean: list = [
        (domain_k, np.mean(corpus_embeddings[df_gb.get_group(domain_k).index], axis=0)) for domain_k in df_gb.size().keys() if domain != domain_k
    ]
    result_find_closed_category: list = [(domain2, np.dot(d_star, em2) / (np.linalg.norm(d_star) * np.linalg.norm(em2))) for domain2, em2 in result_each_category_embedding_mean]
    domain2, result_cosine_domain2 = max(result_find_closed_category, key=lambda a: a[1])
    print("result of finding closed category (domain, domain2, similarity):", domain, domain2, result_cosine_domain2)

    # %%

    cols = "domain,domain2,d_hat_id,d_star_id,result_cosine_domain2"
    l_result.append(eval(cols))

# %%
dfresult = pd.DataFrame(l_result, columns=cols.split(','))
