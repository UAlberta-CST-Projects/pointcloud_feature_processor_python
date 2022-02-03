"""
line surface intersection experiments
"""
import warnings
import numpy as np
from math import sqrt

np.seterr(all='warn')
warnings.filterwarnings('error')


def getclosestpoint(pt, consts):
    """
    fast closest point estimate on an implicit quadric surface using line intersections
    :param pt: an initial pt estimate
    :param consts: the coefficients defining the surface
    :return: a pt as a numpy array
    """
    lines = {'n1_itsc': None, 'n2_itsc': None, 'n3_itsc': None, 'n4_itsc': None}
    lines['n1_itsc'] = getitsc(consts, pt, 'x')
    lines['n2_itsc'] = getitsc(consts, pt, 'y')
    lines['n3_itsc'] = getitsc(consts, pt, 'z')
    if len([line for line in lines.values() if line is not None]) >= 2:
        direction = getaveragedirection(lines, pt)
        lines['n4_itsc'] = getitsc(consts, pt, 'avg', v=direction)

    if lines['n4_itsc'] is not None:
        return lines['n4_itsc']
    else:
        values = [line for line in lines.values() if line is not None]
        if len(values) < 1:
            return pt
        else:
            return values[0]


def getitsc(ct, pt, dim, v=None):
    """
    solve surface equation for single variable, algebra done ahead of time so all the computer has to do
    is crunch numbers directly.
    :param ct: list of constants for the surface equation
    :param pt: the estimated point close to the surface
    :param dim: which variable to solve for (x, y, z, or avg for n4)
    :param v: required when solving for n4, a direction vector
    :return: a point lying on the surface
    """
    # a b c e f g l m n d
    # 0 1 2 3 5 4 6 7 8 9
    # pt = [p1, p2, p3] in expressions, access with pt[0], pt[1], pt[2]
    # v = [v1 v2 v3] in expressions, access with v[0], v[1], v[2]
    if dim == 'x':
        a = ct[0]  # ax^2
        b = 2*ct[3]*pt[1] + 2*ct[4]*pt[2] + 2*ct[6]  # (2ep2 + 2gp3 +2l)x
        c = ct[1]*pt[1]**2 + ct[2]*pt[2]**2 + 2*ct[5]*pt[1]*pt[2] + 2*ct[7]*pt[1] + 2*ct[8]*pt[2] + ct[9]  # bp2^2 + cp3^2 + 2fp2p3 + 2mp2 + 2np3 + d
    elif dim == 'y':
        a = ct[1]  # by^2
        b = 2*ct[3]*pt[0] + 2*ct[5]*pt[2] + 2*ct[7]  # (2ep1 + 2fp3 +2m)y
        c = ct[0]*pt[0]**2 + ct[2]*pt[2]**2 + 2*ct[4]*pt[2]*pt[0] + 2*ct[6]*pt[0] + 2*ct[8]*pt[2] + ct[9]  # ap1^2 + cp3^2 + 2gp1p3 + 2lp1 + 2np3 + d
    elif dim == 'z':
        a = ct[2]  # cz^2
        b = 2*ct[5]*pt[1] + 2*ct[4]*pt[0] + 2*ct[8]  # (2fp2 + 2gp1 +2n)z
        c = ct[0]*pt[0]**2 + ct[1]*pt[1]**2 + 2*ct[3]*pt[0]*pt[1] + 2*ct[6]*pt[0] + 2*ct[7]*pt[1] + ct[9]  # ap1^2 + bp2^2 + 2ep1p2 + 2lp1 + 2mp2 + d
    elif dim == 'avg':
        a = ct[0]*v[0]**2 + ct[1]*v[1]**2 + ct[2]*v[2]**2 + 2*ct[3]*v[0]*v[1] + 2*ct[5]*v[1]*v[2] + 2*ct[4]*v[0]*v[2]  # (av1^2 + bv2^2 + cv3^2 + 2ev1v2 + 2fv2v3 + 2gv1v3)t^2
        b = 2*ct[0]*pt[0]*v[0] + 2*ct[1]*pt[1]*v[1] + 2*ct[2]*pt[2]*v[2] + 2*ct[3]*pt[0]*v[1] + 2*ct[3]*pt[1]*v[0] \
        + 2*ct[5]*pt[1]*v[2] + 2*ct[5]*pt[2]*v[1] + 2*ct[4]*pt[2]*v[0] + 2*ct[4]*pt[0]*v[2] + 2*ct[6]*v[0] + 2*ct[7]*v[1] +2*ct[8]*v[2]  # (2ap1v1 + 2bp2v2 + 2cp3v3 + 2ep1v2 + 2ep2v1 + 2fp2v3 + 2fp3v2 + 2gp3v1 + 2gp1v3 + 2lv1 + 2mv2 + 2nv3)t
        c = ct[0]*pt[0]**2 + ct[1]*pt[1]**2 + ct[2]*pt[2]**2 + 2*ct[3]*pt[0]*pt[1] + 2*ct[5]*pt[1]*pt[2] \
        + 2*ct[4]*pt[0]*pt[2] + 2*ct[6]*pt[0] + 2*ct[7]*pt[1] + 2*ct[8]*pt[2] + ct[9]  # ap1^2 + bp2^2 + cp3^2 + 2ep1p2 + 2fp2p3 + 2gp1p3 + 2lp1 + 2mp2 +2np3 +d
    else:
        raise Exception(f"{dim} is not a valid dim")

    # compute roots with quadratic equation
    roots = []
    itsc = None
    hasroots = False
    try:
        roots.append((-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a))
        hasroots = True
    except:
        pass
    try:
        roots.append((-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a))
        hasroots = True
    except:
        pass

    # handle solutions if they exist
    if hasroots:
        if dim == 'x':
            newpt1 = np.array([roots[0], pt[1], pt[2]])
        elif dim == 'y':
            newpt1 = np.array([pt[0], roots[0], pt[2]])
        elif dim == 'z':
            newpt1 = np.array([pt[0], pt[1], roots[0]])
        else:
            newpt1 = pt + roots[0] * v
        if len(roots) == 2:
            if dim == 'x':
                newpt2 = np.array([roots[1], pt[1], pt[2]])
            elif dim == 'y':
                newpt2 = np.array([pt[0], roots[1], pt[2]])
            elif dim == 'z':
                newpt2 = np.array([pt[0], pt[1], roots[1]])
            else:
                newpt2 = pt + roots[1] * v
            if np.linalg.norm(pt - newpt1) < np.linalg.norm(pt - newpt2):
                itsc = newpt1
            else:
                itsc = newpt2
        else:
            itsc = newpt1
    return itsc


def getaveragedirection(lines, pt):
    """
    simple vector average
    :param lines: a line represented as a point, implicitly originating from the pt argument
    :param pt: a point from which the lines originate
    :return: new direction vector
    """
    avg = np.zeros(3)
    n = 0
    for line in lines:
        if lines[line] is not None:
            vec = lines[line] - pt
            avg += vec
            n += 1
    avg /= n
    return avg
