import numpy as np
from math import ceil


def replicator(A, x, inds, tol, max_iter):
    error = tol + 1.0
    count = 0
    while error > tol and count < max_iter:
        x_old = np.copy(x)
        for i in inds:
            x[i] = x_old[i] * (A[i] @ x_old)
        x /= np.sum(x)
        error = np.linalg.norm(x - x_old)
        count += 1
    return x


def dominant_sets(graph_mat, max_k=0, tol=1e-5, max_iter=1000):
    graph_cardinality = graph_mat.shape[0]
    if max_k == 0:
        max_k = graph_cardinality
    clusters = np.zeros(graph_cardinality)
    already_clustered = np.full(graph_cardinality, False, dtype=np.bool)

    for k in range(max_k):
        if graph_cardinality - already_clustered.sum() <= ceil(0.05 * graph_cardinality):
            break
        # 1000 is added to obtain more similar values when x is normalized
        # x = np.random.random_sample(graph_cardinality) + 1000.0
        x = np.full(graph_cardinality, 1.0)
        x[already_clustered] = 0.0
        x /= x.sum()

        y = replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
        cluster = np.where(y >= 1.0 / (graph_cardinality * 1.5))[0]
        already_clustered[cluster] = True
        clusters[cluster] = k
    clusters[~already_clustered] = k
    return clusters
