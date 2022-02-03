from math import sqrt, log
from numpy.linalg import norm
import numpy as np
from util import planeFit
from multiprocessing import cpu_count, shared_memory
from itertools import repeat
from tqdm import tqdm
import warnings
from surface_intersection import getclosestpoint

np.seterr(all='warn')
warnings.filterwarnings('error')


def init_pool(sha_mem_name, shape, sh_dtype, sel_features):
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
    global selected_features
    shm = shared_memory.SharedMemory(name=sha_mem_name)
    data = np.ndarray(shape, dtype=sh_dtype, buffer=shm.buf)
    selected_features = sel_features


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
    nn = nn.astype('int32')
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
    if len(pts) < 3:
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
    nn = nn.astype('int32')
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
    if len(pts) < 3:
        return np.nan
    # get the plane of best fit as a point and normal vector
    pt, normal = planeFit(data[pts])
    normal = normal / norm(normal)  # normalize the vector
    plane_to_point = data[current_pt] - pt
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
    print("Beginning density work")
    # query tree for nearest neighbors
    if precise:
        dd, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    else:
        dd, nn = tree.query(pc, distance_upper_bound=radius, k=2, workers=-1)
    nn = nn.astype('int32')
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
    nn = nn.astype('int32')
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
    :return: the max height difference found in pts as a foot measurement
    """
    if len(pts) < 2:
        return np.nan
    heights = data[pts]
    heights = heights[:, 2]
    hmax = np.max(heights)
    hmin = np.min(heights)
    return abs(hmax - hmin)


def compute_verticality(pc, tree, PPEexec, radius=0.2, k=20):
    """
    given a 3d point cloud compute the verticality of each point
    :param pc: a dataframe containing the xyz portion of the pointcloud
    :param tree: a nearest neighbors tree for all the points
    :param PPEexec: a ProcessPoolExecutor for crunching all the numbers in parallel
    :param radius: the radius in which to search for nearest neighbors
    :param k: the max number of nearest neighbors to query
    :return: a list of verticality values with the same length as the number of points
    """
    print("Beginning verticality work")
    # query tree for nearest neighbors
    _, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    nn = nn.astype('int32')
    print("determining arg lists...")
    pt_groups = []
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
    print(f"finished creating args, computing verticality for {len(nn)} points")
    verticality = list(tqdm(PPEexec.map(_compute_vert, pt_groups, chunksize=len(pt_groups) // cpu_count()),
                            total=len(pt_groups)))
    return verticality


def _compute_vert(pts):
    '''
    Helper function that computes the verticality of a single point. Built to work with the map function.
    This function determines the verticality of a point by finding the normal of the plane created by the
    surrounding points, then finding the angle between this normal and the positive z-axis (0, 0, 1). Then the angle is
    divided by pi/2 to get a ratio between this normal and the z-axis.
    :param pts: a list of indices for points in the data matrix
    :return: the verticality of the point
    '''
    if len(pts) < 3:
        return np.nan
    # get the plane of best fit as a point and normal vector
    pt, normal = planeFit(data[pts])
    normal = normal / norm(normal)  # normalize the vector
    if normal[2] < 0:
        normal[2] = normal[2]*(-1)
    # angle = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    # xb, yb, zb = (0, 0, 1)
    angle = np.arccos(normal[2]/(np.sqrt(np.square(normal[0]) + np.square(normal[1]) + np.square(normal[2])))) # in radians
    vert = angle/(np.pi/2)
    return vert


def compute_mean_curvature(pc, tree, PPEexec, radius=0.2, k=20):
    """
    given a 3d point cloud compute the mean curvature of each point
    :param pc: a dataframe containing the xyz portion of the pointcloud
    :param tree: a nearest neighbors tree for all the points
    :param PPEexec: a ProcessPoolExecutor for crunching all the numbers in parallel
    :param radius: the radius in which to search for nearest neighbors
    :param k: the max number of nearest neighbors to query
    :return: a list of mean curvature values with the same length as the number of points
    """
    print("Beginning mean curvature work")
    # query tree for nearest neighbors
    dd, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    nn = nn.astype('int32')
    print("determining arg lists...")
    pt_groups = []
    pt_list = []
    pt_idx = 0
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
        pt_list.append(pt_idx)
        pt_idx += 1
    print(f"finished creating args, computing mean curvature for {len(nn)} points")
    mean_curvature = np.array(list(tqdm(PPEexec.map(_compute_m_curve, pt_groups, pt_list, chunksize=len(pt_groups) // cpu_count()), total=len(pt_list))))
    return np.log((mean_curvature - np.nanmin(mean_curvature)) + 1)


def _compute_m_curve(pts, current_pt):
    '''
    Helper function that computes the mean curvature of the plane surrounding a point by finding the equation of the
    plane of best fit, then finding the partial derivatives needed to compute H
    :param pts: a list of indices for points in the data matrix
    :param current_pt: the current point (x, y, z)
    :return: the mean curvature of the point
    '''
    if len(pts) < 6:
        return np.nan
    curve_points = np.array(data[pts])
    current_pt = data[current_pt]
    constants = quadric_equation(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2])
    estimate_pt = getclosestpoint(current_pt, constants)
    k_values = compute_fundamentals(estimate_pt, constants)
    m_curvature = abs((k_values[0] + k_values[1]) / 2)  # H
    return m_curvature


def curve_equation(curve_points, c0, c1, c2, c3, c4, c5):
    # unpacking the multi-dim. array column-wise, that's why the transpose
    x, y, z = curve_points.T
    return (c0 * x ** 2) + (c1 * y ** 2) + (c2 * x * y) + (c3 * x) + (c4 * y) + c5


def compute_gaussian_curvature(pc, tree, PPEexec, radius=0.2, k=20):
    """
    given a 3d point cloud computes the Gaussian curvature of each point
    :param pc: a dataframe containing the xyz portion of the pointcloud
    :param tree: a nearest neighbors tree for all the points
    :param PPEexec: a ProcessPoolExecutor for crunching all the numbers in parallel
    :param radius: the radius in which to search for nearest neighbors
    :param k: the max number of nearest neighbors to query
    :return: a list of Gaussian curvature values with the same length as the number of points
    """
    print("Beginning Gaussian curvature work")
    # query tree for nearest neighbors
    dd, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    nn = nn.astype('int32')
    print("determining arg lists...")
    pt_groups = []
    pt_list = []
    pt_idx = 0
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
        pt_list.append(pt_idx)
        pt_idx += 1
    print(f"finished creating args, computing Gaussian curvature for {len(nn)} points")
    gauss_curvature = np.array(list(tqdm(PPEexec.map(_compute_g_curve, pt_groups, pt_list, chunksize=len(pt_groups) // cpu_count()), total=len(pt_list))))
    return np.log((gauss_curvature - np.nanmin(gauss_curvature)) + 1)


def _compute_g_curve(pts, current_pt):
    '''
    Helper function that computes the Gaussian curvature of the plane surrounding a point by finding the equation of the
    plane of best fit, then finding the partial derivatives needed to compute K
    :param pts: a list of indices for points in the data matrix
    :param current_pt: the current point (x, y, z)
    :return: the Gaussian curvature of the point
    '''
    if len(pts) < 6:
        return np.nan
    curve_points = np.array(data[pts])
    current_pt = data[current_pt]
    constants = quadric_equation(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2])
    estimate_pt = getclosestpoint(current_pt, constants)
    k_values = compute_fundamentals(estimate_pt, constants)
    g_curvature = np.real(k_values[0] * k_values[1])  # K
    return abs(g_curvature * 100)


def quadric_equation(X, Y, Z):
    """
    https://github.com/rbv188/quadric-curve-fit/blob/master/quadrics.py
    :param X: array of x coordinates
    :param Y: array of y coordinates
    :param Z: array of z coordinates
    :return: the 9 equation coefficients
    """
    X = X.reshape((len(X), 1))
    Y = Y.reshape((len(Y), 1))
    Z = Z.reshape((len(Z), 1))
    num = len(X)

    matrix = np.hstack((X**2, Y**2, Z**2, 2*X*Y, 2*X*Z, 2*Y*Z, 2*X, 2*Y, 2*Z))
    output = np.ones((num, 1))

    [a, b, c, d, e, f, g, h, i] = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(matrix), matrix)),
                                                np.transpose(matrix)), output)

    a = -a[0]
    b = -b[0]
    c = -c[0]
    d = -d[0]
    e = -e[0]
    f = -f[0]
    g = -g[0]
    h = -h[0]
    i = -i[0]
    j = 1

    constants = np.array([a, b, c, d, f, e, g, h, i, j])
    #                     0  1  2  3  4  5  6  7  8  9
    return constants


def compute_fundamentals(point, constants):
    # ax^2 + by^2 + cz^2 + exy + fyz + gxz + lx + my + nz + d = 0
    # 0      1      2      3     5     4     6    7    8    9
    Fx = 2*constants[0]*point[0] + constants[3]*point[1] + constants[4]*point[2] + constants[6]
    Fy = 2*constants[1]*point[1] + constants[3]*point[0] + constants[5]*point[2] + constants[7]
    Fz = 2*constants[2]*point[2] + constants[5]*point[1] + constants[4]*point[0] + constants[8]
    Fxx = 2*constants[0]
    Fyy = 2*constants[1]
    Fzz = 2*constants[2]
    Fxy = constants[3]
    Fyz = constants[5]
    Fxz = constants[4]
    grad_F = sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)

    # from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.98.7059&rep=rep1&type=pdf
    E = 1 + (Fx**2 / Fz**2)
    F = Fx * Fy / Fz**2
    G = 1 + (Fy**2 / Fz**2)
    L = (1 / (Fz**2 * grad_F)) * np.linalg.det(np.array(([Fxx, Fxz, Fx], [Fxz, Fzz, Fz], [Fx, Fz, 0])))
    M = (1 / (Fz**2 * grad_F)) * np.linalg.det(np.array(([Fxy, Fyz, Fy], [Fxz, Fzz, Fz], [Fx, Fz, 0])))
    N = (1 / (Fz**2 * grad_F)) * np.linalg.det(np.array(([Fyy, Fyz, Fy], [Fyz, Fzz, Fz], [Fy, Fz, 0])))

    A = np.array(([L, M], [M, N]))
    B = np.array(([E, F], [F, G]))
    B_inv = np.linalg.inv(B)

    k_values = np.linalg.eigvals(np.dot(B_inv, A))
    return k_values


def compute_geometric(pc, tree, PPEexec, sf, radius=0.2, k=20):
    """
    Computes per point eigen features
    :param pc: a dataframe containing the xyz portion of the pointcloud
    :param tree: a nearest neighbors tree for all the points
    :param PPEexec: a ProcessPoolExecutor for crunching all the numbers in parallel
    :param sf: a list of selected features (ints corresponding to features)
    :param radius: the radius in which to search for nearest neighbors
    :param k: the max number of nearest neighbors to query
    :return: a dictionary of computed feature value lists
    """
    print("Beginning feature work")
    # query tree for nearest neighbors
    _, nn = tree.query(pc, distance_upper_bound=radius, k=k, workers=-1)
    nn = nn.astype('int32')
    print("determining arg lists...")
    pt_groups = []
    for knn in nn:
        knn = knn[knn != tree.n]
        pt_groups.append(knn)
    print(f"finished creating args, computing features for {len(nn)} points")
    results_arr = np.array(list(tqdm(PPEexec.map(_compute_geo, pt_groups, chunksize=len(pt_groups) // cpu_count()),
                      total=len(pt_groups))))
    names = {0: "sum", 1: "omnivariance", 2: "eigenentropy", 3: "anisotropy", 4: "planarity", 5: "linearity", 6: "surface_variation", 7: "sphericity", 8: "verticality"}
    results = {}
    for i in range(len(sf)):
        results[names[sf[i]]] = results_arr[:, i]
    return results


def _compute_geo(pts):
    """
    a helper function that computes eigen based geometric features for a single point. Determines a covariance matrix
    for the given points and computes the corresponding eigen values and vectors. The properties of the matrix are such
    that there are 3 eigen values and vectors with the values being non-negative. The following feature calculations
    are simple and are taken from the related paper.
    :param pts: a set of point indices associated with the nearest neighbors
    :return: an array of feature values for the point
    """
    if len(pts) < 4:  # the covariance matrix doesn't work if len(pts) < 3 where 3 represents the number of dimensions
        return np.full(len(selected_features), np.nan)
    # compute covariance matrix
    cov_tensor = np.cov(data[pts], rowvar=False, bias=True)
    # get eigen stuff
    eigvals, eigvects = np.linalg.eig(cov_tensor)
    assert len(eigvals) == 3 and len(eigvects) == 3, "eigen values or vectors not as expected"
    eigvects = np.array([ev for _, ev in sorted(zip(eigvals, eigvects), key=lambda pair: pair[0], reverse=True)])
    eigvals.sort()
    eigvals = np.flip(eigvals)
    assert eigvals[0] >= eigvals[1] >= eigvals[2] >= 0, f"eigen values are not as expected \n{eigvals}\n{cov_tensor}"
    # compute features
    # 0 = sum           5 = linearity
    # 1 = omnivariance  6 = surface variation
    # 2 = eigenentropy  7 = sphericity
    # 3 = anisotropy    8 = verticality
    # 4 = planarity
    results = np.zeros(len(selected_features), dtype=np.float32)
    ind = 0
    for s in selected_features:
        try:
            if s == 0:
                results[ind] = eigvals.sum(dtype=np.float32)
                ind += 1
            elif s == 1:
                results[ind] = eigvals.prod() ** (1/3)
                ind += 1
            elif s == 2:
                lneig = np.log(eigvals)
                results[ind] = -np.dot(lneig, eigvals)
                ind += 1
            elif s == 3:
                results[ind] = (eigvals[0] - eigvals[2]) / eigvals[0]
                ind += 1
            elif s == 4:
                results[ind] = (eigvals[1] - eigvals[2]) / eigvals[0]
                ind += 1
            elif s == 5:
                results[ind] = (eigvals[0] - eigvals[1]) / eigvals[0]
                ind += 1
            elif s == 6:
                results[ind] = eigvals[2] / eigvals.sum(dtype=np.float32)
                ind += 1
            elif s == 7:
                results[ind] = eigvals[2] / eigvals[0]
                ind += 1
            elif s == 8:
                results[ind] = 1 - abs(np.dot(np.array([0,0,1]), eigvects[2]))
                ind += 1
        except RuntimeWarning:
            results[ind] = np.nan
            ind += 1
    return results

