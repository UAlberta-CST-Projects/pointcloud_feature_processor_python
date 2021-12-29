import numpy as np

from scipy.spatial import cKDTree
from collections import deque


def euclidean_cluster_extraction(pts: np.ndarray, min_distance: float) -> np.ndarray:
    """
    Clustering as described in section 6.2 of "Semantic 3D Object Maps for Everyday Manipulation in Human Living
    Environments" (http://mediatum.ub.tum.de/doc/800632/941254.pdf). This is the same as what the PCL calls "Euclidean
    Cluster Extraction" and should also be equivalent to MATLAB's "pcsegdist".
    """

    print("Number of points passed to the cKDTree (pts.size): " + str(pts.size))
    tree = cKDTree(pts, balanced_tree=False, compact_nodes=False)
    q = deque()
    clusters = []
    processed = np.zeros(pts.shape[0], dtype=bool)

    for i in range(pts.shape[0]):
        if processed[i]:
            continue

        q.clear()
        q.append(i)
        cluster = []
        processed[i] = True
        while q:
            j = q.popleft()
            cluster.append(j)
            for n in tree.query_ball_point(pts[j], min_distance):
                if not processed[n]:
                    processed[n] = True
                    q.append(n)
        clusters.append(cluster)

    cluster_ids = np.zeros(pts.shape[0], dtype=np.int)
    for cluster_id, cluster in enumerate(clusters):
        for cluster_pt_idx in cluster:
            cluster_ids[cluster_pt_idx] = cluster_id

    return cluster_ids
