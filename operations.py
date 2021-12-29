from math import sqrt
from scipy.spatial import cKDTree
from numpy.linalg import norm
import numpy as np
from util import planeFit
from multiprocessing import cpu_count, shared_memory
from itertools import repeat
from tqdm import tqdm


def init_pool(sha_mem_name, shape, sh_dtype):
    global data
    global shm  # fun fact: if this isn't global the entire memory block offs itself, I'd like my day back please.
    shm = shared_memory.SharedMemory(name=sha_mem_name)
    data = np.ndarray(shape, dtype=sh_dtype, buffer=shm.buf)
    #print(data)


def compute_gradient(pc, tree, PPEexec, gfield='z', radius=0.2, k=20):
    """
    given a 3d point cloud compute the gradient over the x, y, or z coordinate
    :param pc: a dataframe containing the xyz portion of the pointcloud
    :param tree: a nearest neighbors tree for all the points
    :param PPEexec: a ProcessPoolExecutor for crunching all the numbers in parallel
    :param gfield: the coordinate you want to compute gradient over
    :param radius: the radius in which to search for nearest neighbors
    :param k: the max number of nearest neighbors to query
    :return: a list of slopes with the same length as the number of points
    """
    _, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    # for each point get the related nearest neighbors and compute gradient
    print("determining arg lists...")
    pt_groups = []
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
    print(f"finished creating args, computing gradient for {len(nn)} points")
    slopes = list(tqdm(PPEexec.map(_compute_grad, pt_groups, repeat(gfield), chunksize=len(pt_groups) // cpu_count()), total=len(pt_groups)))
    return slopes


def _compute_grad(pts, gfield):
    if len(pts) < 4:
        return np.nan
    # get the points as a numpy array
    # get the best fit plane as a point and normal vector
    pt, normal = planeFit(data[pts])
    normal = normal / norm(normal)  # normalize the vector
    # depending on the field selected calculate the appropriate slope magnitude
    if gfield.lower() == 'z':
        if normal[2] < 0:
            normal -= 2 * normal
        # the distance of the normal to the vertical is the slope, this is the magnitude of the x, y coords for z gradient
        slope = sqrt(normal[0] ** 2 + normal[1] ** 2)
    elif gfield.lower() == 'x':
        if normal[0] < 0:
            normal -= 2 * normal
        slope = sqrt(normal[1] ** 2 + normal[2] ** 2)
    elif gfield.lower() == 'y':
        if normal[1] < 0:
            normal -= 2 * normal
        slope = sqrt(normal[0] ** 2 + normal[2] ** 2)
    else:
        raise Exception('gfield must be x, y, or z')
    return slope


def compute_roughness(pc, tree, PPEexec, radius=0.2, k=20):
    """
    given a 3d point cloud compute the roughness of each point
    :param pc: a dataframe containing the xyz portion of the pointcloud
    :param tree: a nearest neighbors tree for all the points
    :param PPEexec: a ProcessPoolExecutor for crunching all the numbers in parallel
    :param radius: the radius in which to search for nearest neighbors
    :param k: the max number of nearest neighbors to query
    :return: a list of roughness values with the same length as the number of points
    """
    # build nearest neighbor tree
    _, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    print("determining arg lists...")
    pt_groups = []
    pt_list = []
    pt_idx = 0
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(pc.iloc[knn].to_numpy())
        pt_list.append(pc.iloc[pt_idx].to_numpy())
        pt_idx += 1
    print(f"finished creating args, computing roughness for {len(nn)} points")
    roughnesses = list(tqdm(PPEexec.map(_compute_rough, pt_groups, pt_list, chunksize=len(pt_groups) // cpu_count()), total=len(pt_list)))
    return roughnesses


def _compute_rough(pts, current_pt):
    if len(pts) < 4:
        return np.nan
    # get the plane of best fit as a point and normal vector
    pt, normal = planeFit(pts)
    normal = normal / norm(normal)  # normalize the vector
    plane_to_point = current_pt - pt
    dist_to_plane = np.dot(normal, plane_to_point) * normal  # project plane to point vector onto plane normal
    dist_to_plane = norm(dist_to_plane)
    return dist_to_plane


def compute_density(pc, tree, PPEexec, radius=0.2, precise=False, k=20):
    """
    given a 3d point cloud compute the density around each point.
    :param pc: a dataframe containing the xyz portion of the pointcloud
    :param tree: a nearest neighbors tree for all the points
    :param PPEexec: a ProcessPoolExecutor for crunching all the numbers in parallel
    :param radius: the radius in which to search for nearest neighbors
    :param precise: indicates whether density is reported as number of neighbors (True), or as a rougher distance based approximation (False)
    :param k: the max number of nearest neighbors to query
    :return: a list of density values with the same length as the number of points
    """
    if precise:
        dd, nn = tree.query(pc.xyz, distance_upper_bound=radius, k=k, workers=-1)
    else:
        dd, nn = tree.query(pc.xyz, distance_upper_bound=radius, k=2, workers=-1)
    print("determining arg lists...")
    pt_groups = []
    dd_groups = []
    dd_group = 0
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
        dd_groups.append(dd[dd_group])
        dd_group += 1
    print(f"finished creating args, computing density for {len(nn)} points")
    densities = list(tqdm(PPEexec.map(_compute_den, pt_groups, dd_groups, repeat(radius), repeat(precise), chunksize=len(pt_groups) // cpu_count()), total=len(pt_groups)))
    return densities


def _compute_den(pts, dd, radius, precise):
    if len(pts) <= 0:
        return np.nan
    if precise:
        return len(pts)
    else:
        # use distance to the nearest neighbor as an inverse measure of density
        if len(pts) < 2:
            # the closest neighbor should theoretically be the point itself, so check and choose accordingly
            if dd[0] <= 0.01:
                return 0
            else:
                return 1 - (dd[0] / radius)
        else:
            return 1 - (dd[1] / radius)
