import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import pdist, cdist
from scipy.linalg import svd

class Aligner2D:
    """
    Aligns 2D spatial data points across time points using Procrustes analysis and point permutation.
    """

    def __init__(self, well_name="Well_Data", time_points=None):
        """
        Initializes the Aligner2D class.

        Args:
            well_name (str): Name or identifier of the well (for plotting and saving).
            time_points (list): List of labels representing time points. Default is range(number of time points)
        """
        self.well_name = well_name
        self.time_points = time_points
        self.aligned_points = None  # Stores aligned points
        self.best_orders = None    # Stores best permutation orders
        self.errors = None        # Stores alignment errors for each time point

    def _center_points(self, points):
        """Centers a set of 2D points around the origin (0, 0)."""
        return points - np.mean(points, axis=0)

    def _compute_distance_matrix(self, points, metric='sqeuclidean'):
        """Computes the pairwise distance matrix between points."""
        return cdist(points, points, metric=metric)

    def _find_best_permutation(self, pointsA, pointsB, metric='sqeuclidean'):
        """Finds the permutation of pointsB minimizing difference with pointsA"""
        N = pointsA.shape[0]
        min_error = np.inf
        best_order = None
        dist_matrix_a = self._compute_distance_matrix(pointsA, metric=metric)

        for perm in itertools.permutations(range(N)):
            permuted_pointsB = pointsB[perm, :]
            dist_matrix_b = self._compute_distance_matrix(permuted_pointsB, metric=metric)
            error = np.linalg.norm(dist_matrix_a - dist_matrix_b)

            if error < min_error:
                min_error = error
                best_order = perm
        return best_order, min_error

    def _procrustes_align(self, pointsA, pointsB):
        """Performs Procrustes alignment, handling 1D point arrays."""
        if pointsA.ndim == 1:
            pointsA = pointsA.reshape(-1, 1)  # Reshape to column vector
        if pointsB.ndim == 1:
            pointsB = pointsB.reshape(-1, 1)

        centroidA = np.mean(pointsA, axis=0)
        centroidB = np.mean(pointsB, axis=0)
        centeredA = pointsA - centroidA
        centeredB = pointsB - centroidB

        if centeredA.shape[1] == 1 or centeredB.shape[0] == 1:
            return pointsA, np.eye(2), centroidA - centroidB # return a workable transform if dimensions mismatch for 1 point

        H = centeredB.T @ centeredA
        U, _, Vt = svd(H)  # No need for .conj().T with SVD
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroidA - centroidB @ R
        aligned_pointsB = (pointsB @ R) + t

        return aligned_pointsB, R, t

    def align(self, dataX, dataY, well_id, **kwargs):
        """
        Aligns points over time relative to the first time point.
        """
        data_0 = np.vstack((dataX[well_id, 0], dataY[well_id, 0])).T
        data_0 = self._center_points(data_0)

        self.aligned_points = [data_0] # initialize aligned points with the first timepoint
        self.best_orders = [] # best order at each timepoint
        self.errors = []  # Alignment errors for each time point
        all_registered_data = [] # for building pandas dataframe

        time_points = kwargs.get("time_points", list(range(dataX.shape[1])))

        for j in range(1, dataX.shape[1]): 
            data_k = np.vstack((dataX[well_id, j], dataY[well_id, j])).T
            data_k = self._center_points(data_k)

            pointsA = data_0.astype(np.float64)
            pointsB = data_k.astype(np.float64)

            best_order, min_error = self._find_best_permutation(pointsA, pointsB)
            self.best_orders.append(best_order)
            self.errors.append(min_error)
            ordered_pointsB = pointsB[best_order]

            aligned_pointsB, rotation_matrix, translation_vector = self._procrustes_align(pointsA, ordered_pointsB)
            self.aligned_points.append(aligned_pointsB)

            for raw_x, raw_y, reg_x, reg_y in zip(
                dataX[well_id, j], dataY[well_id, j], aligned_pointsB[:, 0], aligned_pointsB[:, 1]
            ):
                all_registered_data.append(
                    {
                        'raw_x': raw_x, 
                        'raw_y': raw_y, 
                        'registered_x': reg_x, 
                        'registered_y': reg_y, 
                        'timepoint': time_points[j]
                    }
                )

        self.registered_df = pd.DataFrame(all_registered_data) #added dataframe property
        
        if kwargs.get('plot', False):
            self.plot(**kwargs) #call plotting function if needed

        return self

    def plot(self, fig=None, axs=None, title_legend=None, marker_shape=None, marker_colors=None, save=False, save_dir=None, show_plot=True, **kwargs):
        """Plots the aligned data."""
        if title_legend is None:
            title_legend = self.time_points
        n_cols = len(title_legend) -1
        if n_cols == 0: #if only one time point exists, then just plot the initial points and the aligned points using a single subplot.
            fig, axs = plt.subplots(1, 1, figsize = (6,3)) #create a single subplot if there are no other time points other than the initial one.
            pointsA = self.aligned_points[0] #always relative to the first timepoint
            pointsB = self.aligned_points[1] if len(self.aligned_points) > 1 else None
            if pointsB is not None:
                axs.scatter(pointsA[:, 0], pointsA[:, 1], marker='o', c='blue', label='Initial Points')
                axs.scatter(pointsB[:, 0], pointsB[:, 1], marker='x', c='red', label='Aligned Points')
                axs.set_xticks([])
                axs.set_yticks([])
                axs.legend(loc='best')
                fig.suptitle(f'{self.well_name}: Aligned Point Sets', fontsize=14)  # Updated suptitle
                plt.tight_layout()
            else:
                axs.scatter(pointsA[:, 0], pointsA[:, 1], marker='o', c='blue', label='Initial Points')
                axs.set_xticks([])
                axs.set_yticks([])
                axs.legend(loc='best')
                fig.suptitle(f'{self.well_name}: Aligned Point Sets', fontsize=14)  # Updated suptitle
                plt.tight_layout()

            #fig, axs = plt.subplots(2, len(self.aligned_points) - 1, figsize=(12, 6))

        elif (fig is not None) and (axs is not None): #check if axes and figures are provided
            fig = fig
            axs = axs

        # handle the case where the correct number of time points is provided.
        else:  
            fig, axs = plt.subplots(2, len(self.aligned_points) - 1, figsize=(12, 6))

        if marker_shape is None:
            marker_shape = ['o', '^', 's', 'd', '*'] 
        if marker_colors is None:
            marker_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if title_legend is None:
            title_legend = [f"Time {i+1}" for i in range(len(self.aligned_points) - 1)] #adjust the title legend

        # Main plotting loop (now using aligned_points list directly)
        for j in range(len(self.aligned_points) - 1):
            pointsA = self.aligned_points[0] #always relative to the first timepoint
            pointsB = self.aligned_points[j+1]

            # Plotting logic (similar to your original code) –– updated axs[0, j-1] --> axs[0, j]
            axs[0, j].scatter(pointsA[:, 0], pointsA[:, 1], marker=marker_shape[0], c=marker_colors[0], label='Points A (Original Polygon)')
            axs[0, j].scatter(pointsB[:, 0], pointsB[:, 1], marker=marker_shape[j+1], c=marker_colors[j+1], label=f'Points B (Time {j+1})')
            axs[0, j].set_xticks([]) # updated from axs[0, j-1].set_xticks([])
            axs[0, j].set_yticks([])  # updated from axs[0, j-1].set_yticks([])
            axs[0, j].set_title(title_legend[j], fontsize=10) #adjust title
            axs[0, j].legend(loc='best', fontsize=6) #add legend for each subplot

            axs[1, j].scatter(pointsA[:, 0], pointsA[:, 1], marker=marker_shape[0], c=marker_colors[0], label='Points A (Original Polygon)')
            axs[1, j].scatter(pointsB[:, 0], pointsB[:, 1], marker=marker_shape[j+1], c=marker_colors[j+1], label=f'Aligned Points B (Time {j+1})')
            axs[1, j].plot(np.append(pointsA[:, 0], pointsA[0, 0]), np.append(pointsA[:, 1], pointsA[0, 1]), marker_colors[0], linewidth=2.0, alpha=0.6)
            axs[1, j].plot(np.append(pointsB[:, 0], pointsB[0, 0]), np.append(pointsB[:, 1], pointsB[0, 1]), marker_colors[j+1], alpha=1.0, linewidth=1.0)
            axs[1, j].set_xticks([])
            axs[1, j].set_yticks([])
            axs[1, j].legend(loc='best', fontsize=6) #add legend for each subplot

        fig.suptitle(f'{self.well_name}: Aligned Point Sets', fontsize=14) #updated suptitle
        plt.tight_layout()
        if save:
            save_dir = save_dir or 'alignment_output'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, f"{self.well_name}_aligned_points.png"))  # Updated filename
        if show_plot:
            plt.show()









##########
def convert_raw_to_long(df, **kwargs):
    df.columns = [x.lower() for x in df.columns]
    og_point_name = kwargs.get('point_name', 'original point name')
    timepoints = kwargs.get('timepoints', ["0h", "24h", "48h", "72h", "168h"])
    long_data = []
    for _, row in df.iterrows():
        point_name = row[og_point_name]
        for timepoint in timepoints:
            x_col = f'x_{timepoint}'
            y_col = f'y_{timepoint}'
            
            if x_col in df.columns and y_col in df.columns:  # Check if columns exist
                x_val = row[x_col]
                y_val = row[y_col]
    
                long_data.append(
                    {
                        'Original Point Name': point_name,
                        'Timepoint': timepoint,
                        'raw_x': x_val,
                        'raw_y': y_val
                    }
                )
    
    # Create the long-format DataFrame
    long_df = pd.DataFrame(long_data)
    return long_df

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

def match_points(raw_data, well_id, registered_df, best_orders, **kwargs):
    opts = {
        'point_name_0h':'label_0h',
        'x_0h':'x_0h',
        'y_0h':'y_0h'
    }
    opts.update(kwargs)
    
    raw_well_data = raw_data.iloc[well_id,:]

    point_names = raw_well_data[[opts['point_name_0h'], opts['x_0h'], opts['y_0h']]]
    point_names.columns = ['point_name', 'x', 'y']
    point_ordering = [point_names['point_name'].values[bo].astype('unicode') for bo in best_orders]
    point_ordering = [x for points in point_ordering for x in points]
    registered_df['Aligned Point Name'] = point_ordering
    
    raw_well_data = raw_well_data.rename(columns = {'label_0h' : 'Original Point Name'})
    raw_well_data_long = convert_raw_to_long(raw_well_data)
    merged_data = pd.merge(raw_well_data_long, registered_df, on = ['raw_x', 'raw_y'])
    matched_data = merged_data[['Original Point Name', 'Aligned Point Name', 'Timepoint', 'raw_x','raw_y','registered_x','registered_y','timepoint']]
    time_diff = matched_data.pop('timepoint')
    matched_data.insert(3, 'Time-Diff', time_diff)
    # sort the dataframe such that the aligned points are in order
    matched_data = matched_data.sort_values(by=['Aligned Point Name', 'Timepoint'])

    return matched_data