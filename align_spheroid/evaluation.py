# align_spheroid/evaluation.py (New module)
import numpy as np
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes

def relative_movement(path1, path2):
    """
    Calculates the relative movement of points in path2 compared to their corresponding 
    points in path1, normalized by the distance to the nearest neighbor in path1.

    Args:
        path1 (np.ndarray): 2D array of points representing the first path (reference path).
        path2 (np.ndarray): 2D array of points representing the second path.

    Returns:
        np.ndarray: 1D array of relative movements, expressed as percentages.
                     Returns None if the paths have different numbers of points.
                     Returns None if either path is empty.
    """

    if len(path1) != len(path2):
        print("Paths must have the same number of points for relative movement calculation.")
        return None
    
    if len(path1) == 0 or len(path2) == 0:
        print("Paths cannot be empty.")
        return None

    if len(path1) < 2: # nearest neighbor search can not happen on paths of only one point.
        print("Paths must have at least two points to perform nearest neighbor search.")
        return None

    distances_path1 = cdist(path1, path1)  # Pairwise distances within path1

    # Exclude distance to self for NN search. Setting diagonal to large value
    np.fill_diagonal(distances_path1, np.inf)

    nearest_neighbor_distances_path1 = np.min(distances_path1, axis=1)

    relative_movements = []
    for i in range(len(path2)):
        corresponding_distance = np.linalg.norm(path2[i] - path1[i])
        if nearest_neighbor_distances_path1[i] == 0: # handle if nearest points overlap
            relative_movement_percent = 0
        else:
            relative_movement_percent = (corresponding_distance / nearest_neighbor_distances_path1[i]) * 100
        relative_movements.append(relative_movement_percent)

    return np.array(relative_movements)

def turning_function(path):
    """Calculates the turning function of a path. Handles edge cases gracefully.

    Args:
        path (np.ndarray): A 2D NumPy array where each row represents a point (x, y).

    Returns:
        np.ndarray: A 1D NumPy array representing the turning function. 
                     Returns an empty array if the path has fewer than 3 points.
    """

    if len(path) < 3:
        return np.array([]) #return empty if it has less than 3 points

    angles = []
    for i in range(len(path) - 2):
        v1 = path[i+1] - path[i]
        v2 = path[i+2] - path[i+1]

        #avoid division by zero error handling
        if np.all(v1 == 0) or np.all(v2 == 0): #if both are zero there's no change in direction
            angle = 0.0
        elif np.dot(v1,v2) == 0: # handle division by zero case where cos(angle) = 0. 
            angle = np.pi/2 if np.cross(v1,v2) > 0 else -np.pi/2 # calculate angle of rotation based on cross product.
        else: #standard calculation
            angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))

        angles.append(angle)

    return np.cumsum(angles)


def frechet_distance(p, q):
    """
    Computes the discrete Fréchet distance between two curves, represented as sequences of points. The Fréchet distance is a measure of similarity between curves that takes into account the order of the points.

    Args:
        p (np.ndarray): A 2D numpy array representing the first curve. Each row is a point (x, y).
        q (np.ndarray): A 2D numpy array representing the second curve. Each row is a point (x, y).

    Returns:
        float: The discrete Fréchet distance between the two curves.
    """
    ca = np.ones((len(p), len(q))) * -1 # Initialize a matrix to store computed distances

    ca[0, 0] = np.linalg.norm(p[0] - q[0]) # Base case: distance between the first points

    # Fill the first column
    for i in range(1, len(p)):
        ca[i, 0] = max(ca[i-1, 0], np.linalg.norm(p[i] - q[0]))

    # Fill the first row
    for j in range(1, len(q)):
        ca[0, j] = max(ca[0, j-1], np.linalg.norm(p[0] - q[j]))
    
    # Dynamic programming to compute remaining distances
    for i in range(1, len(p)):
        for j in range(1, len(q)):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]), np.linalg.norm(p[i] - q[j]))

    return ca[len(p)-1, len(q)-1] # Return the distance between the endpoints

def calculate_frechet(path1, path2):
    """
    Calculates the Fréchet distance between two paths.

    The Fréchet distance measures the similarity between two curves or paths by considering the order of the points. 

    Args:
        path1 (np.ndarray): A 2D NumPy array representing the first path.
        path2 (np.ndarray): A 2D NumPy array representing the second path.

    Returns:
        float: The Fréchet distance between the two paths. Returns np.inf if either path is None.
    """

    if path1 is None or path2 is None:
        print("Cannot calculate Frechet distance with empty paths")
        return np.inf

    return frechet_distance(path1, path2)


def compare_paths(path1, path2):
    """Compares two paths using Fréchet distance of turning functions. Handles edge cases.

    Args:
        path1: (np.ndarray): A 2D NumPy array representing the first path.
        path2: (np.ndarray): A 2D NumPy array representing the second path.


    Returns:
        float: Fréchet distance. Returns np.inf if paths are invalid.
    """
    if path1 is None or path2 is None or len(path1) < 3 or len(path2) < 3:
        return np.inf

    tf1 = turning_function(path1)
    tf2 = turning_function(path2)

    if tf1.size == 0 or tf2.size == 0: # if any of the turning functions are empty
        return np.inf #return infinity if empty

    if tf1.shape[0] < 1 or tf2.shape[0] < 1: #handle case where paths are too short or malformed after calling turning_function()
       return np.inf

    return calculate_frechet(tf1, tf2)

def procrustes_distance(pointsA, pointsB):
    """Calculates Procrustes distance, handling 1D cases gracefully."""

    mtx1 = np.array(pointsA, dtype=np.double, copy=True)
    mtx2 = np.array(pointsB, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.shape[0] <= mtx1.shape[1]: #if more columns than rows or equal number of rows and columns
       mtx1 = mtx1.T #transpose
    if mtx2.shape[0] <= mtx2.shape[1]:
        mtx2 = mtx2.T

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    mtx1 /= norm1
    mtx2 /= norm2

    # Handle cases where orthogonal_procrustes returns 2 or 3 values
    try:
        _, _, disparity = orthogonal_procrustes(mtx1, mtx2)
    except ValueError:
        _, disparity = orthogonal_procrustes(mtx1, mtx2) #get only two outputs

    return disparity

def hausdorff_distance(pointsA, pointsB):
    """Calculates the Hausdorff distance between two point sets.

    Args:
        pointsA (np.ndarray): First point set.
        pointsB (np.ndarray): Second point set.

    Returns:
        float: Hausdorff distance.
    """
    return max(directed_hausdorff(pointsA, pointsB)[0], directed_hausdorff(pointsB, pointsA)[0])


def earth_movers_distance(pointsA, pointsB):
    """Calculates the Earth Mover's Distance (EMD) between two point sets.
       Also known as the Wasserstein distance. Uses the Euclidean distance as the ground metric.

    Args:
        pointsA (np.ndarray): First point set.
        pointsB (np.ndarray): Second point set.

    Returns:
        float: Earth Mover's Distance.
    """

    if len(pointsA) == 0 or len(pointsB) == 0: #if either set is empty
        return np.inf

    n, m = len(pointsA), len(pointsB)
    cost_matrix = cdist(pointsA, pointsB)  # Calculate cost matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Use Hungarian algorithm
    emd = cost_matrix[row_ind, col_ind].sum() / min(n,m) # normalize by the number of points
    return emd