from math import sqrt
from numpy.linalg import norm
import numpy as np
from util import planeFit
from multiprocessing import cpu_count, shared_memory
from itertools import repeat
from tqdm import tqdm


def init_pool(sha_mem_name, shape, sh_dtype):
    """
    Runs at the start of every worker process spawned by ProcessPoolExecutor.
    Provides the worker access to the pointcloud data via shared memory.
    :param sha_mem_name: The name of a created SharedMemory buffer, used to attach to it.
    :param shape: The shape of the numpy ndarray containing the point data.
    :param sh_dtype: The dtype of the numpy ndarray containing the point data.
    :return: None
    """
    global data
    global shm  # fun fact: if this isn't global the entire memory block offs itself, I'd like my day back please.
    shm = shared_memory.SharedMemory(name=sha_mem_name)
    data = np.ndarray(shape, dtype=sh_dtype, buffer=shm.buf)


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
    print("Beginning gradient work")
    # query tree for nearest neighbors
    _, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    print("determining arg lists...")
    pt_groups = []
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
    print(f"finished creating args, computing gradient for {len(nn)} points")
    # for each point get the related nearest neighbors and compute gradient
    slopes = list(tqdm(PPEexec.map(_compute_grad, pt_groups, repeat(gfield), chunksize=len(pt_groups) // cpu_count()), total=len(pt_groups)))
    return slopes


def _compute_grad(pts, gfield):
    """
    Helper function that computes the gradient of a single point. Built to work with the map function.
    This function finds the slope around a point by fitting a plane to the surrounding points.
    From this, a point and normal vector that define the plane are determined and used to calculate slope.
    :param pts: The indices of the points within the specified radius of the current point.
    :param gfield: x, y, or z, determines on which plane gradient is computed for.
    :return: The slope value of the given area determined by the given points (the gradient).
    """
    if len(pts) < 4:
        return np.nan
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
    print("Beginning roughness work")
    # query tree for nearest neighbors
    _, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    print("determining arg lists...")
    pt_groups = []
    pt_list = []
    pt_idx = 0
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
        pt_list.append(pt_idx)
        pt_idx += 1
    print(f"finished creating args, computing roughness for {len(nn)} points")
    roughnesses = list(tqdm(PPEexec.map(_compute_rough, pt_groups, pt_list, chunksize=len(pt_groups) // cpu_count()), total=len(pt_list)))
    return roughnesses


def _compute_rough(pts, current_pt):
    """
    Helper function that computes the roughness of single point. Built to work with the map function.
    This function determines 'roughness' of a point by calculating how far the point is from the plane
    of best fit of the surrounding points.
    :param pts: The indices of the points within the specified radius of the current point.
    :param current_pt: The point that roughness is currently being calculated for.
    :return: The distance from the current point to the plane of best fit (the roughness).
    """
    if len(pts) < 4:
        return np.nan
    # get the plane of best fit as a point and normal vector
    pt, normal = planeFit(data[pts])
    normal = normal / norm(normal)  # normalize the vector
    plane_to_point = data[current_pt] - pt
    dist_to_plane = np.dot(normal, plane_to_point) * normal  # project plane to point vector onto plane normal
    #dist_to_plane = norm(dist_to_plane)
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
    print("Beginning density work")
    # query tree for nearest neighbors
    if precise:
        dd, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    else:
        dd, nn = tree.query(pc, distance_upper_bound=radius, k=2, workers=-1)
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
    """
    Helper function that computes the density of single point. Built to work with the map function.
    This function determines the density surrounding a point by either counting the number of neighbouring points
    within the specified radius, or by using the distance to the nearest point as an inverse estimation of the density.
    :param pts: The indices of the points within the specified radius of the current point. The length of this list
    is the only real information necessary.
    :param dd: A list of the distances to the nearest neighbors which align with those in pts.
    :param radius: The radius that was specified to be considered for density.
    :param precise: Indicates whether density is reported as number of neighbors (True),
    or as a rougher distance based approximation (False).
    :return: The density of the current point as either the number of nearest neighbors or a distance approximation.
    """
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


def compute_max_local_height_difference(pc, tree, PPEexec, radius=0.2, k=20):
    """
    given a 3d point cloud compute the max height difference between the points around every point
    :param pc: a dataframe containing the xyz portion of the pointcloud
    :param tree: a nearest neighbors tree for all the points
    :param PPEexec: a ProcessPoolExecutor for crunching all the numbers in parallel
    :param radius: the radius in which to search for nearest neighbors
    :param k: the max number of nearest neighbors to query
    :return: a list of height difference values with the same length as the number of points
    """
    print("Beginning height diff work")
    # query tree for nearest neighbors
    _, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    print("determining arg lists...")
    pt_groups = []
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
    print(f"finished creating args, computing height diff for {len(nn)} points")
    diffs = list(tqdm(PPEexec.map(_compute_height_diff, pt_groups, chunksize=len(pt_groups) // cpu_count()), total=len(pt_groups)))
    return diffs


def _compute_height_diff(pts):
    """
    Helper function that computes the height difference around a single point. Built to work with the map function.
    This function determines the max z difference in an area around a point by getting the difference between
    the min and max z coordinates from the list of points passed.
    :param pts: a list of indices for points in the data matrix
    :return: the max height difference found in pts as a centimeter measurement
    """
    if len(pts) < 2:
        return np.nan
    heights = data[pts]
    heights = heights[:, 2]
    hmax = np.max(heights)
    hmin = np.min(heights)
    return abs(hmax - hmin) * 100  # return diff in cm, assuming coords are in m
