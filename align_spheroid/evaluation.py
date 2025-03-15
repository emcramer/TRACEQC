# align_spheroid/evaluation.py (New module)
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes

def alignment_accuracy(simulated_df, registered_df, time_points):
    """Calculates the alignment accuracy at each time point.

    Args:
        simulated_df (pd.DataFrame): DataFrame from CellSimulator.
        registered_df (pd.DataFrame): DataFrame from Aligner2D.
        time_points (list): List of time point labels.

    Returns:
        np.ndarray: Array of accuracy fractions for each time point.
    """

    n_timepoints = len(time_points)
    accuracies = []

    for i in range(n_timepoints): #modified to work across timepoints.
        time = time_points[i]
        # Extract labels from simulated data
        true_labels = simulated_df[f'label_{time}' if i > 0 else 'label'].tolist()  # Handle "label" at time 0
        # Map registered labels to their original order
        registered_at_time = registered_df[registered_df['timepoint'] == time] if i > 0 else registered_df.copy()
        if i > 0:
            label_map = dict(zip(registered_df['label'], simulated_df['label']))
            registered_labels = [label_map[label] for label in registered_at_time['label']]
        else: #no registered labels at time point 0
            registered_labels = registered_at_time['label']
        
        # Sort both label lists to compare correctly
        true_labels.sort()
        registered_labels.sort()
        # Calculate accuracy for the current time point
        n_correct = sum(t == r for t, r in zip(true_labels, registered_labels))

        accuracy = n_correct / len(true_labels) if len(true_labels) > 0 else 1.0  # Handle empty time point
        accuracies.append(accuracy)

    return np.array(accuracies)

def normalize_distance(path, distance):
    """
    Calculates a normalized version of a distance metric between two paths. Divides the distance metric by the minimum nearest neighbor distance in the provided path.

    Args:
        path (np.ndarray): 2D array of points representing the first path (reference path). 
        distance (np.float): Scalar value indicating the measured distance between two paths.

    Returns:
        np.float: A scalar value indicating the distance between two paths normalized to the minimum nearest neighbor distance of the initial path.
    """
    if len(path) < 2:
        return float('inf')
    distances = distance_matrix(path, path)
    # Exclude distances to self (diagonal elements)
    distances[np.diag_indices_from(distances)] = np.inf
    return distance/np.min(distances)

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

def distance_matrix_error(points1, points2, metric='euclidean'):
    """Calculates the mean squared error (MSE) between two distance matrices.

    Args:
        points1 (np.ndarray): First set of points (2D array).
        points2 (np.ndarray): Second set of points (2D array).
        metric (str, optional): Distance metric to use. Defaults to 'euclidean'.

    Returns:
        float: Mean squared error between the distance matrices.
              Returns np.inf if either point set is empty or if they have different numbers of points.
    """

    if points1.size == 0 or points2.size == 0: # if either point set is empty, return np.inf. This behavior can be modified as needed.
        return np.inf

    if points1.shape[0] != points2.shape[0]:
        return np.inf  # Handle cases with different numbers of points


    dist_matrix1 = cdist(points1, points1, metric=metric)
    dist_matrix2 = cdist(points2, points2, metric=metric)

    mse = np.mean((dist_matrix1 - dist_matrix2)**2)

    return mse

def build_long_true(data, n, timepoints):
    x_cols = [c for c in data.columns if "x" in c]
    y_cols = [c for c in data.columns if "y" in c]
    label_cols = [c for c in data.columns if "label" in c]
    
    xvals = data[x_cols].values.T.ravel()
    yvals = data[y_cols].values.T.ravel()
    tplabels = data[label_cols].values.T.ravel()
    tps = [j for subl in [[i]*n for i in timepoints] for j in subl]
    
    long_true = pd.DataFrame({
        'true_label':tplabels,
        'timepoint':tps,
        'raw_x':xvals,
        'raw_y':yvals
    })
    return long_true

def calc_accuracy(true_labels, pred_labels):
    n = len(true_labels)
    true_preds = true_labels == pred_labels
    n_true = sum(true_preds)
    accp = n_true/n *100
    return accp, true_preds