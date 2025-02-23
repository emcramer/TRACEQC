import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.spatial import procrustes
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid
import itertools
from collections import namedtuple
from typing import Tuple, Dict

# create a named tuple for returning data
PolygonSimilarity = namedtuple("PolygonSimilarity", ['iou', 'dice', 'hausdorff', 'frechet_distance'])

def turning_function_path(path):
    """
    Calculates the turning function of a path.

    The turning function represents the cumulative change in direction as you move along the path.
    It's a useful shape descriptor because it's invariant to translation, rotation, and scaling.

    Args:
        path (np.ndarray): A 2D NumPy array where each row represents a point (x, y) along the path.

    Returns:
        np.ndarray: A 1D NumPy array representing the turning function of the path.
                     The array's length is two less than the original path because of difference calculations.
                     Returns None if the path has fewer than 3 points.
    """
    n = len(path)
    if n < 3:  # Need at least 3 points to calculate angles
        print("Path must have at least 3 points to calculate the turning function.")
        return None

    angles = []
    for i in range(n - 2):
        v1 = path[i+1] - path[i]
        v2 = path[i+2] - path[i+1]
        angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
        angles.append(angle)

    turning_function = np.cumsum(angles)
    return turning_function

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

def compare_path_turning_functions(path1, path2):
    """
    Compares two paths by calculating the Fréchet distance between their turning functions.

    This method provides a shape-based comparison that is invariant to translation, rotation, and scaling.

    Args:
        path1 (np.ndarray): A 2D NumPy array representing the first path.
        path2 (np.ndarray): A 2D NumPy array representing the second path.

    Returns:
        float: The Fréchet distance between the turning functions of the two paths.
               Returns np.inf if either turning function cannot be calculated (e.g., path has < 3 points).
    """
    tf1 = turning_function_path(path1)
    tf2 = turning_function_path(path2)


    distance = calculate_frechet(tf1, tf2)  # Use helper function for clarity.

    return distance

def turning_function(polygon):
    """
    Calculates the turning function of a polygon.  The turning function represents the cumulative change in direction 
    as you traverse the perimeter of the polygon.  It's a useful representation for shape comparison because it's invariant
    to translation and scaling.

    Args:
        polygon (np.ndarray): A 2D numpy array where each row represents a vertex (x, y) of the polygon.

    Returns:
        np.ndarray: A 1D numpy array representing the turning function of the polygon.
    """
    angles = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]  # Wrap around to the first vertex for the last edge
        p3 = polygon[(i + 2) % len(polygon)]
        v1 = p2 - p1  # Vector from p1 to p2
        v2 = p3 - p2  # Vector from p2 to p3
        angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2)) # Angle between v1 and v2
        angles.append(angle)
    turning = np.cumsum(angles)  # Cumulative sum of angles gives the turning function
    return turning

def align_polygons(polygonA, polygonB):
    """
    Aligns two polygons using the turning function and Procrustes analysis. This function finds the optimal cyclic permutation of polygonB's
    vertices that minimizes the Fréchet distance between the turning functions of polygonA and polygonB. Then it refines the alignment using Procrustes analysis.

    Args:
        polygonA (np.ndarray): A 2D numpy array representing the first polygon (reference polygon).
        polygonB (np.ndarray): A 2D numpy array representing the second polygon (polygon to be aligned).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The aligned polygonB after Procrustes analysis.
            - list: The indices of the best permutation found using the turning function.
            - float: The minimum Fréchet distance found between the turning functions.
    """
    best_order = None
    min_distance = np.inf

    tf_A = turning_function(polygonA) # Compute the turning function of polygonA

    # Iterate over every permutation of polygonB's indices
    for perm in itertools.permutations(range(len(polygonB))):
        permuted_polygonB = polygonB[list(perm)]
        tf_B = turning_function(permuted_polygonB)
        distance = frechet_distance(tf_A, tf_B)

        if distance < min_distance:
            min_distance = distance
            best_order = list(perm)

    # Use Procrustes analysis to refine the alignment using the least squares method
    mtx1, mtx2, disparity = procrustes(polygonA, polygonB[best_order])

    return mtx2, best_order, min_distance  # Aligned polygonB


def polygon_similarity(polygon1, polygon2, tolerance=1e-6):
    """
    Calculates IoU, Dice, and Hausdorff distance between two polygons,
    handling self-intersections.

    Args:
        polygon1: Shapely Polygon or MultiPolygon (or numpy array of coordinates).
        polygon2: Shapely Polygon or MultiPolygon (or numpy array of coordinates).
        tolerance (float, optional): Precision tolerance for very small areas.

    Returns:
        tuple: (IoU, Dice, Hausdorff). Returns None if any polygon is empty or invalid.
    """
    # Convert numpy arrays to shapely objects, if needed
    if type(polygon1) is np.ndarray:
        polygon1 = Polygon(polygon1)
    if type(polygon2) is np.ndarray:
        polygon2 = Polygon(polygon2)

    # check to make sure that the polygons are proper
    if polygon1.is_empty or polygon2.is_empty or not polygon1.is_valid or not polygon2.is_valid: # Check if polygon1 and polygon2 are not empty or invalid
        print(f"Invalid or empty polygons detected.") # more descriptive error message
    if polygon1.is_empty:
        print("Polygon 1 is empty.")
    if not polygon1.is_valid:
        print("Polygon 1 is invalid.")
    if polygon2.is_empty:
        print("Polygon 2 is empty.")
    if not polygon2.is_valid:
        print("Polygon 2 is invalid.")    
    return None  # Return None to signal an error
        
    # Handle potential invalid polygons (self-intersections). 
    poly1 = polygonize(polygon1.exterior) # Decomposes self-intersections
    polygon1 = unary_union(list(poly1))

    poly2 = polygonize(polygon2.exterior)
    polygon2 = unary_union(list(poly2))

    if polygon1.is_empty or polygon2.is_empty:  # Check if they're empty or invalid
        return None
    
    # Handle MultiPolygons: Calculate metrics across component polygons, if necessary.
    intersection_area = 0
    for p1 in getattr(polygon1, 'geoms', [polygon1]):  # Iterates over single or multi polygons.
        for p2 in getattr(polygon2, 'geoms', [polygon2]):
            intersection_area += p1.intersection(p2).area

    union_area = polygon1.union(polygon2).area
    iou = intersection_area / union_area if union_area > tolerance else 0
    dice = (2 * intersection_area) / (polygon1.area + polygon2.area) if (polygon1.area + polygon2.area) > tolerance else 0
    hausdorff = directed_hausdorff(polygon1, polygon2)[0]
    return iou, dice, hausdorff

def alignpoly2d(dataX, dataY):
    # assuming only the data for the well is passed with x and y positions as individual 1D arrays
    # Initialize aligned data storage
    data_0 = np.vstack((dataX[0], dataY[0])).T
    data_0 = data_0 - np.mean(data_0, axis=0)  # Centered around zero

    # for each of the time points in the data, perform alignment with the polygon
    # number of timepoints should be the length of the position arrays, skipping the first timepoint but including the last
    n_timepoints = len(dataX) 
    for k in range(1, n_timepoints+1): 
        data_k = np.vstack((dataX[k], dataY[k])).T
        data_k = data_k - np.mean(data_k, axis=0)  # Centered around zero

        pointsA = data_0.astype(np.float64)
        pointsB = data_k.astype(np.float64)
    
        # Example Usage (adapting your code):
        # create the "polygons" after centering data_0 and data_k ...
        polygonA = pointsA
        polygonB = pointsB

        print("polygon A:\n")
        print(polygonA)
        print("\n-----\n")
        print("polygon B:\n")
        print(polygonB)
        
        aligned_polygonB, best_order, frechet_dist = align_polygons(polygonA, polygonB)
    
        sim_results = polygon_similarity(polygonA, aligned_polygonB)

        if sim_results is not None: # Check if the result is not None (indicating an error in the inputs).
            iou, dice, hausdorff = sim_results  # Unpack only if the result is valid.
            print(f"IoU: {iou}")
            print(f"Dice: {dice}")
            print(f"Hausdorff Distance: {hausdorff}")
            print(f"Frechet Distance between Turning Functions: {frechet_dist}")
            
            # return the measurements of polygon similarity
            return PolygonSimilarity(iou, dice, hausdorff, frechet_dist)
        else:
            print("Polygon similarity calculation failed due to invalid or empty polygons.")
            return None # return None and allow function to continue (or handle the error appropriately).

def align2d(dataX, dataY, well_id, well_name, **kwargs):
    """
    Aligns time-series spatial data points using Procrustes analysis and permutation to minimize
    the difference between original and transformed points.

    Parameters:
    -----------
    dataX : numpy.ndarray
        2D array representing the X-coordinates of data points for different wells across time points. 
        Each row corresponds to a well, and each column corresponds to a time point.

    dataY : numpy.ndarray
        2D array representing the Y-coordinates of data points for different wells across time points. 
        Each row corresponds to a well, and each column corresponds to a time point.

    well_id : int
        Index of the specific well to be analyzed.

    well_name : str
        Name or identifier of the well, used for labeling plots.
    Returns:
    --------
    reg_dataX : list
        A list containing the X-coordinates of the aligned points for each time point.
        
    reg_dataY : list
        A list containing the Y-coordinates of the aligned points for each time point.

    Functionality:
    --------------
    1. **Initial Setup**:
       - The function begins by selecting the data points at the first time point (index 0) for the specified well. 
       - These points are centered by subtracting the mean of the coordinates, so that they are centered at the origin.

    2. **Iterative Alignment**:
       - The function iterates through subsequent time points (`dataX[:, j]` and `dataY[:, j]` for `j=1,...,n-2`), performing the following steps for each:
         
         a. **Re-centering**: 
            - The data points are centered in the same manner as the points from time point 0.
         
         b. **Distance Matrix Computation**:
            - The pairwise distance matrix is computed for both the original (`data_0`) and current time points using the Euclidean distance metric.
         
         c. **Point Permutation**:
            - All possible permutations of the points in the current time point are generated. 
            - For each permutation, the distance matrix is calculated, and the permutation that minimizes the error (difference between distance matrices) is chosen.
         
         d. **Alignment via Procrustes**:
            - Procrustes analysis is performed, which computes the optimal transformation (rotation and translation) to align the reordered points to the original points. This is done by:
                - Computing the centroids of both sets of points.
                - Centering the points by subtracting their centroids.
                - Using Singular Value Decomposition (SVD) to calculate the optimal rotation matrix.
                - Ensuring that the rotation matrix is proper (i.e., a reflection is avoided).
                - Applying the rotation and translation to align the points.
         
    3. **Plotting**:
       - The function generates plots to visualize the original and aligned data points for each time point:
         - **Top Row**: Shows the original points and the transformed (before alignment) points.
         - **Bottom Row**: Shows the aligned points compared to the original points.
    
    4. **Return**:
       - The aligned data points for all time points (`reg_dataX` and `reg_dataY`) are returned as lists.

    Example:
    --------
    reg_dataX, reg_dataY = align(dataX, dataY, well_id=3, well_name='Well 3')

    This will align the time series data for `well_id` 3, plot the results, and return the aligned X and Y coordinates.
    """

    # Initialize aligned data storage
    data_0 = np.vstack((dataX[well_id, 0], dataY[well_id, 0])).T
    data_0 = data_0 - np.mean(data_0, axis=0)  # Centered around zero

    # Store results in a list of dictionaries, easier for DataFrame creation
    all_registered_data = []
    
    reg_dataX = [data_0[:, 0]]
    reg_dataY = [data_0[:, 1]]
    title_legend = kwargs.get('timepoints', ['0h-24h', '0h-48h', '0h-72h', '0-168h'])
    n_timepoints = len(title_legend)

    aligned_points = []
    best_orders = []
    ordered_points = []
    timepoint_errors = []

    if (kwargs.get('fig', None) is not None) and (kwargs.get('ax', None) is not None):
        fig = kwargs['fig']
        axs = kwargs['axs']
    else:
        fig, axs = plt.subplots(2, len(title_legend), figsize=(12, 6))
    
    # Define marker shapes for each time point
    marker_shape = ['o', '^', 's', 'd', '*'] 
    marker_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for j in range(1, n_timepoints+1): #for j in range(1, dataX.shape[1]-1):
        data_k = np.vstack((dataX[well_id, j], dataY[well_id, j])).T
        data_k = data_k - np.mean(data_k, axis=0)  # Centered around zero

        pointsA = data_0.astype(np.float64)
        pointsB = data_k.astype(np.float64)

        # Plot the original and transformed points
        axs[0, j-1].scatter(pointsA[:, 0], pointsA[:, 1], marker=marker_shape[0], c=marker_colors[0], label='Points A (Original Polygon)')
        axs[0, j-1].scatter(pointsB[:, 0], pointsB[:, 1], marker=marker_shape[j], c=marker_colors[j], label='Points B (Transformed and Shuffled)')
        plt.grid(True)
        axs[0,j-1].set_xticks([])
        axs[0,j-1].set_yticks([])
        axs[0, j-1].set_title(title_legend[j-1], fontsize=18)

        dist_metric = kwargs.get('dist_metric', 'sqeuclidean')

        # Compute the pairwise distance matrices
        distMatrixA = cdist(pointsA, pointsA, dist_metric)
        #distMatrixB = cdist(pointsB, pointsB, dist_metric)

        # Reorder pointsB to match pointsA
        N = pointsA.shape[0] # number of points
        bestOrder = None
        minError = np.inf 

        all_permutations = np.array(list(itertools.permutations(range(N))))
        for perm in all_permutations:
            permutedPointsB = pointsB[perm, :]
            permutedDistMatrixB = cdist(permutedPointsB, permutedPointsB, dist_metric)
            error = np.linalg.norm(distMatrixA - permutedDistMatrixB, 2)

            if error < minError:
                minError = error
                bestOrder = perm
                #print(bestOrder, minError)
        #print(f"Minimum Erorr Across Timepoints for {well_name}")
        #print(minError)
        orderedPointsB = pointsB[bestOrder, :]
        best_orders.append(bestOrder)
        ordered_points.append(orderedPointsB)
        timepoint_errors.append(minError)

        # Compute centroids
        centroidA = np.mean(pointsA, axis=0)
        centroidB = np.mean(orderedPointsB, axis=0)

        #print(f"Centroid A: {centroidA}, Centroid B: {centroidB}")

        # Center the points by subtracting their centroids
        centeredA = pointsA - centroidA
        centeredB = orderedPointsB - centroidB

        # Compute optimal rotation using SVD
        H = centeredB.conj().T @ centeredA
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T.conj()
        rotationMatrix = V @ U.conj().T

        # Ensure the rotation matrix is proper
        if np.linalg.det(rotationMatrix) < 0:
            V[:, -1] = -V[:, -1]
            rotationMatrix = V @ U.conj().T

        #print(f"Rotation Matrix for {well_name}")
        #print(rotationMatrix)

        # Apply the transformation
        translationVector = centroidA - (centroidB @ rotationMatrix)

        #print("Translation Vector")
        #print(translationVector)

        alignedPointsB = (orderedPointsB @ rotationMatrix.conj().T) + translationVector

        for raw_x, raw_y, reg_x, reg_y in zip(
            dataX[well_id, j], dataY[well_id, j], alignedPointsB[:, 0], alignedPointsB[:, 1]
        ):
            all_registered_data.append(
                {
                    'raw_x': raw_x, 
                    'raw_y': raw_y, 
                    'registered_x': reg_x, 
                    'registered_y': reg_y, 
                    'timepoint': title_legend[j-1]
                }
            )

        #print(f"Aligned Points for {well_name}")
        #print(alignedPointsB)

        # Plot the aligned points
        axs[1, j-1].scatter(pointsA[:, 0], pointsA[:, 1], marker=marker_shape[0], c=marker_colors[0], label='Points A (Original Polygon)')
        axs[1, j-1].scatter(alignedPointsB[:, 0], alignedPointsB[:, 1], marker=marker_shape[j], c=marker_colors[j], label='Aligned Points B')
        axs[1, j-1].plot(np.append(pointsA[:, 0], pointsA[0, 0]), np.append(pointsA[:, 1], pointsA[0, 1]), marker_colors[0], linewidth=4.0, alpha=0.6)#'b-o')
        axs[1, j-1].plot(np.append(alignedPointsB[:, 0], alignedPointsB[0, 0]), np.append(alignedPointsB[:, 1], alignedPointsB[0, 1]), marker_colors[j], alpha=1.0, linewidth=1.0)#'r-x')
        axs[1,j-1].set_xticks([])
        axs[1,j-1].set_yticks([])
        axs[1, j-1].grid(color=".9")
        #axs[1, j-1].set_title(title_legend[j-1], fontsize=22)

        reg_dataX.append(alignedPointsB[:, 0])
        reg_dataY.append(alignedPointsB[:, 1])
        aligned_points.append(alignedPointsB)

    fig.suptitle(f'{well_name}: Centered Points and Polygon Matching', fontsize=24)
    
    if kwargs.get('save', False) is True: 
        # if a directory was provided, use it to save output
        if kwargs.get('save_dir', None) is not None:
            save_dir = kwargs['save_dir']
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, f"{well_name}_registered_points_and_polygon_matching.png"))
        # otherwise create a directory in the working directory and save output
        else:
            save_dir = 'alignment_output'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, f"{well_name}_registered_points_and_polygon_matching.png"))
    if kwargs.get('show_plot', False):
        plt.show()
    #plt.close()

    # Create the DataFrame outside the loops
    registered_df = pd.DataFrame(all_registered_data)

    #return reg_dataX, reg_dataY, translationVector, rotationMatrix, centroidA, centroidB
    return registered_df, best_orders, ordered_points, reg_dataX, reg_dataY, timepoint_errors