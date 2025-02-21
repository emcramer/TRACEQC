import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import pdist, cdist
from scipy.linalg import svd

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
blob:vscode-webview://0ogvi7tud9mte63pu58mm901lnkack6047f8g0iu1r99u0g21qoq/efb87888-6e68-478c-9510-1defd3487d36
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