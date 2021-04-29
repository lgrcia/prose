import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, product
from scipy.spatial import KDTree

# GEOMETRY
# --------


def ang(x, y):
    "angle between two point x, y with respect to x axis"
    return np.arctan(1/np.divide(*(y-x)))


def reorganize(a, b, c, d, return_idxs=False):
    """
    order coordinates from closest to first one
    """
    x = np.vstack([a, b, c, d])
    distances = np.linalg.norm(x[:, :, None] - x.T[None, :, :], axis=1)
    i = np.min(np.unravel_index(np.argmax(distances), distances.shape))
    A = x[i]
    x = np.delete(x, i, axis=0)
    distances_from_A = np.linalg.norm(x - A, axis=1)
    idxs = np.argsort(1 / distances_from_A)
    
    if return_idxs:
        return [0, *idxs]
    else:
        return [A, *x[idxs]]
    
    
def rotate_point(point, angle, pivot, norm=False):
    """
    rotate point around pivot of certain angle
    """
    co = np.cos(angle); si = np.sin(angle)
    r = np.array([
        [co, -si],
        [si,  co]
    ])
    x = point - pivot
    x = r@x
    if norm:
        x /= np.linalg.norm(x)
        x *= np.linalg.norm(point - pivot)*co
    x += pivot
    
    return x


def XY(a, b, norm=False):
    """
    coordinates of the x, y axis as defined in Lang2009
    """
    if norm:
        norm = np.linalg.norm(b-a)
    x = rotate_point(b, -np.pi/4, a, norm=norm)
    y = rotate_point(b, np.pi/4, a, norm=norm)
    
    return x, y


def proj(p, origin, axe, norm=False):
    """
    projection of a point p on a segment from origin to axe
    """
    n = axe - origin
    n /= np.linalg.norm(n, 2)
    if norm:
        return np.dot(p - origin, n)
    else:
        return origin + n*np.dot(p - origin, n)
    
    
# QUAD
# ----

def quad_hash(a, b, c, d):
    """
    from 4 coordinates froduce the quad hash code
    """
    x, y = XY(a,b)
    h = np.linalg.norm(b-a)
    xd = proj(d, a, x, norm=True)/h; yd = proj(d, a, y, norm=True)/h
    xc = proj(c, a, x, norm=True)/h; yc = proj(c, a, y, norm=True)/h
    return xc, xd, yc, yd


def good_quad(a, b, c, d):
    """
    whether all points are contained in a circle (see Lang2009)
    """
    r = np.linalg.norm(b-a)/2
    center = a + (b-a)/2
    # check distance from center
    in_circle = np.linalg.norm(center - np.vstack([a, b, c, d]), axis=1) <= r*1.01
    return np.all(in_circle)


def clean(xy, tolerance=20):

    distances_to_others = np.array([np.linalg.norm(p-xy, axis=1) for p in xy])
    return xy[np.argwhere(np.sum(distances_to_others < tolerance, axis=0) == 1).flatten()]

# AFFINE TRANSFORM
# ----------------


pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:,:-1]


def _find_transform(s1, s2):
    """
    Strict finding matrix transform between registered points
    least square to find the affine transform matrix between 2 set of points
    """
    S1 = pad(s1)
    S2 = pad(s2)
    A, res, rank, s = np.linalg.lstsq(S1, S2, rcond=None)
    
    return A.T


def affine_transform(M):
    return lambda x: unpad(np.dot(pad(x), M.T))


# PLOTTING
# -------


def plot(*args, color="k", offset=5, label=None, alpha=1, **kwargs):
    """
    Conveniant plot of poitn sources
    """
    for i, a in enumerate(args):
        plt.plot(*a, "o", fillstyle="none", c=color, label=label if i==0 else None, alpha=alpha)
    for i, (name, a) in enumerate(kwargs.items()):
        plt.plot(*a, "o", fillstyle="none", c=color, label=label if i==0 else None, alpha=alpha)
        plt.text(a[0], a[1] + offset, name, ha='center', color=color)
    plt.gca().set_aspect("equal")
    if label is not None:
        plt.legend()
    
    
def plot_quad(a, b, c, d):
    """
    Plot to visualize quad when making hash code, as in Lang2009
    """
    x, y = XY(a, b, norm=True)

    xd = proj(d, a, x); yd = proj(d, a, y)
    xc = proj(c, a, x); yc = proj(c, a, y)

    plot(a=a, b=b, c=c, d=d)
    plot(x=x, y=y, color="C0")
    plot(xd=xd, yd=yd, xc=xc, yc=yc, color="C0")

    plt.plot(*np.array([d, xd]).T, "--", color="C0", alpha=0.2)
    plt.plot(*np.array([d, yd]).T, "--", color="C0", alpha=0.2)
    plt.plot(*np.array([c, xc]).T, "--", color="C0", alpha=0.2)
    plt.plot(*np.array([c, yc]).T, "--", color="C0", alpha=0.2)

    plt.plot(*np.array([a, x]).T, color="C0", alpha=0.2)
    plt.plot(*np.array([a, y]).T, color="C0", alpha=0.2)

    plt.gca().add_patch((plt.Circle((b-a)/2 + a, radius=np.linalg.norm(b-a)/2, fill=False)))
    plt.gca().add_patch((plt.Polygon(np.array([a, c, b, d]), facecolor="k", alpha=0.05)))

    
# Full match
# ----------

def count_cross_match(s1, s2, tolerance=2):
    """
    count pair of points whose distance is less than tolerance
    """
    c = 0
    for i, s in enumerate(s1):
        distances = np.linalg.norm(s - s2, axis=1)
        closest = np.argmin(distances)
        if distances[closest] < tolerance:
            c += 1
    return c


def quads_stars(xy, n=15):
    """
    return matched indexes bewteen two set of points
    """
    xy = xy.copy()
    xy = xy[0:n]

    quads_idxs = list(combinations(np.arange(xy.shape[0]), 4))

    quads = []
    stars = []

    for qi in quads_idxs:
        _quad = reorganize(*xy[qi, :])
        if good_quad(*_quad):
            quads.append(quad_hash(*_quad))
            stars.append(_quad)

    if len(quads) == 0:
        print(len(quads))

    return np.array(quads), np.array(stars)


def cross_match(s1, s2, tolerance=10, return_ixds=False):
    matches = []

    for i, s in enumerate(s1):
        distances = np.linalg.norm(s - s2, axis=1)
        closest = np.argmin(distances)
        if distances[closest] < tolerance:
            matches.append([i, closest])

    matches = np.array(matches)

    if return_ixds:
        return matches
    else:
        if len(matches) > 0:
            return s1[matches[:, 0]], s2[matches[:, 1]]
        else:
            return np.array([]), np.array([])


def find_transform(s1, s2, tolerance=10, n=15, show=False):
    quads1, stars1 = quads_stars(s1, n=n)
    quads2, stars2 = quads_stars(s2, n=n)

    # KDTree
    kdt = KDTree(quads1)
    dist, indices = kdt.query(quads2)

    # We pick the two asterisms leading to the highest stars matching
    closeness = []
    for i, m in enumerate(indices):
        M = _find_transform(stars1[m], stars2[i])
        new_s1 = affine_transform(M)(s1)
        closeness.append(count_cross_match(s2, new_s1, tolerance=tolerance))

    i = np.argmax(closeness)
    m = indices[i]
    S1 = stars1[m]
    S2 = stars2[i]
    M = _find_transform(S1, S2)
    new_s1 = affine_transform(M)(s1)

    if show:
        rs1, rs2 = cross_match(new_s1, s2, tolerance=tolerance)
        plot(*rs1)
        plot(*rs2, color="C3")

    i, j = cross_match(new_s1, s2, tolerance=tolerance, return_ixds=True).T

    return _find_transform(s1[i], s2[j])


# Some optimized methods

def _reorganize(Q):
    distances = np.linalg.norm((Q[:, :, :, None] - np.swapaxes(Q, 1, 2)[:, None, :, :]), axis=2)
    return np.array(
        [Q[i][np.argsort(distances[i, m])[[0, 3, 2, 1]]] for i, m in enumerate(np.argmax(distances, 1)[:, 0])])


def _count_cross_match(s1, s2, tolerance=2):
    """
    count pair of points whose distance is less than tolerance
    """
    return np.count_nonzero(np.linalg.norm(s1[None, :] - s2[:, None], axis=2) < tolerance)


def _good_quad(a, b, c, d, max_distance=1000):
    """
    whether all points are contained in a circle (see Lang2009)
    """
    x = np.vstack([a, b, c, d])
    r = np.linalg.norm(b - a) / 2
    center = a + (b - a) / 2
    # check distance from center
    in_circle = np.linalg.norm(center - x, axis=1) <= r * 1.01
    max_distance = np.max(np.linalg.norm(x[:, :, None] - x.T[None, :, :], axis=1)) < max_distance
    return np.all(in_circle) and max_distance


def _quad_hash(a, b, c, d):
    """
    from 4 coordinates froduce the quad hash code
    """
    x, y = XY(a, b)
    h = np.linalg.norm(b - a)
    n = [x, y] - a
    n = (n / np.linalg.norm(n, 2)).T
    xd, yd = np.dot(d - a, n) / h
    xc, yc = np.dot(c - a, n) / h
    return xc, xd, yc, yd


def _quads_stars(xy, n=15):
    """
    return matched indexes bewteen two set of points
    """
    xy = xy.copy()
    xy = xy[0:n]

    quads_idxs = np.array(list(combinations(np.arange(xy.shape[0]), 4)))

    Q = xy[quads_idxs]
    Q = _reorganize(Q)

    quads = []
    stars = []

    for q in Q:
        if _good_quad(*q):
            quads.append(_quad_hash(*q))
            stars.append(q)

    return quads, stars