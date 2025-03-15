import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import pdist, cdist
from scipy.linalg import svd
from sklearn.cluster import KMeans

class Aligner2D:
    """
    Aligns 2D spatial data points across time points using Procrustes analysis and point permutation.
    """

    def __init__(self, well_name="Well_Data", time_points=None):
        """
        Initializes the Aligner2D class.

        Args:
            well_name (str): Name or identifier of the well (for plotting and saving).
            time_points (list, optional): List of labels representing time points. If None, it defaults to a range.
        """
        self.well_name = well_name
        self.time_points = time_points
        self.aligned_points = None
        self.best_orders = None
        self.errors = None
        self.registered_df = None
        self.centeredA = None
        self.centeredB = []


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
            error = np.linalg.norm(dist_matrix_a - dist_matrix_b, 2)

            if error < min_error:
                min_error = error
                best_order = perm
        return best_order, min_error
    
    def _procrustes_align(self, pointsA, pointsB):
        """Performs Procrustes alignment, handling 1D point arrays."""

        # check dimensionality to make sure a and b match and algorithm can proceed
        if pointsA.ndim == 1:
            pointsA = pointsA.reshape(-1, 1)
        if pointsB.ndim == 1:
            pointsB = pointsB.reshape(-1, 1)

        centroidA = np.mean(pointsA, axis=0)
        centroidB = np.mean(pointsB, axis=0)

        centeredA = pointsA - centroidA
        centeredB = pointsB - centroidB

        # store the centered points
        self.centeredA = centeredA
        self.centeredB.append(centeredB)

        #return pointsA, np.eye(pointsA.shape[1]), np.zeros_like(centroidA) #identity

        if centeredA.shape[1] == 1 or centeredB.shape[0] == 1: #updated to check for 1 point AFTER centering
            return pointsA, np.eye(2), centroidA - centroidB # return a workable transform if dimensions mismatch for 1 point

        # Compute optimal rotation using SVD
        H = centeredB.conj().T @ centeredA
        U, S, Vt = svd(H)  #
        V = Vt.T.conj()
        R = V @ U.conj().T

        # Ensure the rotation matrix is proper
        if np.linalg.det(R) < 0:
            V[:, -1] = -V[:, -1]
            R = V @ U.conj().T
            
        # Apply rotation to translation
        t = centroidA - (centroidB @ R) #updated translation
        aligned_pointsB = (pointsB @ R.conj().T) + t #apply transformation as before
        return aligned_pointsB, R, t

    def align(self, dataX, dataY, labels_0=None, **kwargs):
        """
        Aligns points over time relative to the first time point.

        Args:
            dataX (numpy.ndarray): 2D array of x-coordinates. Shape (n_timepoints, n_points).
            dataY (numpy.ndarray): 2D array of y-coordinates. Shape (n_timepoints, n_points).

        Returns:
            self: Returns the Aligner2D instance for method chaining.
        """

        if dataX.shape != dataY.shape:
            raise ValueError("dataX and dataY must have the same shape.")

        # Determine the number of points and time points from the input data
        n_timepoints = len(self.time_points)
        n_points = dataX.shape[0]
        
        data_0 = np.vstack((dataX[:, 0], dataY[:, 0])).T 
        data_0 = self._center_points(data_0)

        self.aligned_points = [data_0]
        self.best_orders = []
        self.errors = []
        all_registered_data = []
        self.unaligned_points = []  # Store unaligned, centered points

        # Use provided time points or generate a range of indices if none are given
        time_points = self.time_points if self.time_points is not None else list(range(n_timepoints))
        self.time_points = time_points #update time points

        if labels_0 is not None:  # If labels are provided (simulated data), store them
            self.labels = labels_0 #store initial labels
        else: #make generic labels for real data case.
            self.labels = [f'Point_{i}' for i in range(dataX.shape[0])]

        # Store initial labels in registered_df for ALL time points
        for j in range(n_timepoints):  # Iterate through ALL time points, starting at 0
            data_j = np.vstack((dataX[:, j], dataY[:, j])).T  # Combine x and y
            data_j = self._center_points(data_j)
            if j > 0: #only append if timepoint is after time 0.
                self.unaligned_points.append(data_j)
            for k, (raw_x, raw_y) in enumerate(zip(dataX[:, j], dataY[:, j])):
                all_registered_data.append(
                    {
                        'raw_x': raw_x,
                        'raw_y': raw_y,
                        'registered_x': data_j[k, 0],  # Store centered x, y for time 0
                        'registered_y': data_j[k, 1],
                        'timepoint': time_points[j],
                        'original_label': self.labels[k],  # Store initial label
                        'aligned_label': self.labels[k]  # Initialize aligned label (same as original at time 0)

                    }
                )
        
        for j in range(1, n_timepoints):
            data_k = np.vstack((dataX[:, j], dataY[:, j])).T #updated
            data_k = self._center_points(data_k)
            centered_data_k = self._center_points(data_k)  # Center before permutation
            self.unaligned_points.append(centered_data_k)
            
            pointsA = data_0.astype(np.float64)
            pointsB = data_k.astype(np.float64)
            
            best_order, min_error = self._find_best_permutation(pointsA, pointsB)
            self.best_orders.append(best_order)
            self.errors.append(min_error)
            ordered_pointsB = pointsB[best_order, :]

            aligned_pointsB, rotation_matrix, translation_vector = self._procrustes_align(pointsA, ordered_pointsB)
            self.aligned_points.append(aligned_pointsB)

            # Correctly reorder the labels based on best_order for the current timepoint j+1
            current_labels = [self.labels[idx] for idx in best_order] #get corresponding labels
            # Add aligned points and reordered labels to the DataFrame
            
            # Update registered_df with aligned data and labels for current time point j
            current_registered = [data for data in all_registered_data if data['timepoint'] == time_points[j]]
            for k, (raw_x, raw_y, reg_x, reg_y) in enumerate(zip(
                dataX[:, j], dataY[:, j], aligned_pointsB[:, 0], aligned_pointsB[:, 1]
            )):

                current_registered[best_order[k]].update({
                    'registered_x': reg_x,
                    'registered_y': reg_y,
                    'aligned_label': self.labels[k],  # Correctly ordered label
                })

        # Recreate DataFrame after modifications, since changes are not being made in-place.
        self.registered_df = pd.DataFrame(all_registered_data)

        self.aligned_points_df = pd.concat([pd.DataFrame(arr, columns = ['x', 'y']) for arr in self.aligned_points])
        self.registered_df = pd.DataFrame(all_registered_data)
        return self  # Return self for method chaining

    def plot(
        self, 
        fig=None, 
        axs=None, 
        title_legend=None, 
        marker_shape=None, 
        marker_colors=None, 
        save=False, 
        save_dir=None, 
        show_plot=True, 
        **kwargs
    ):
        """Plots the aligned data. If fig and axs are provided they must have the correct number of rows and columns for the number of timepoints.
        Otherwise, an error will be raised and no plots will be created.

        Args:
            fig (matplotlib.figure.Figure, optional): Matplotlib figure. Defaults to None.
            axs (numpy.ndarray of matplotlib.axes.Axes, optional): Array of Matplotlib axes, expects a 2 x (len(self.aligned_points) - 1) array of axes. Defaults to None.
            title_legend (list of str, optional): Legend titles for each time point. Defaults to self.time_points if available, otherwise creates defaults.
            marker_shape (list, optional): List of marker shapes. Defaults to default markers.
            marker_colors (list, optional): List of marker colors. Defaults to default colors.
            save (bool, optional): If True, saves the plot. Defaults to False.
            save_dir (str, optional): Directory to save the plot. Defaults to "alignment_output".
            show_plot (bool, optional): If True, shows the plot. Defaults to True.
        """

        if title_legend is None:
            if self.time_points is not None:
                title_legend = self.time_points
            else:
                title_legend = [f"Time {i+1}" for i in range(len(self.aligned_points) - 1)]
        n_cols = len(title_legend) - 1  # Adjust n_cols for plotting

        # Case where dataX and dataY have the same shape for all time points (no change occurs during alignment).

        if n_cols == 0:
            if (fig is not None) and (axs is not None):
                if not isinstance(axs, np.ndarray): # check if axs is an array
                    axs = np.array([axs]) #make it into a numpy array so we can index correctly.
                # Check dimensions of fig and axs here and raise ValueError if they don't match. For now, we are expecting a single axis
                if axs.size != 1:
                    raise ValueError(f"axs must be a single axis for one time point. Got an array of shape {axs.shape}")
            else:
                fig, axs = plt.subplots(1, 1, figsize=(6, 3))  # Create a single subplot if only one time point besides initial

            # Initialize aligned_points with data_0
            pointsA = self.aligned_points[0] #first timepoint
            pointsB = self.aligned_points[1] if len(self.aligned_points) > 1 else None # second timepoint, or None if it doesn't exist

            if pointsB is not None:
                axs.scatter(pointsA[:, 0], pointsA[:, 1], marker='o', c='blue', label='Initial Points')
                axs.scatter(pointsB[:, 0], pointsB[:, 1], marker='x', c='red', label='Aligned Points')
                axs.set_xticks([])
                axs.set_yticks([])
                axs.legend(loc='best')
                fig.suptitle(f'{self.well_name}: Aligned Point Sets', fontsize=14)
                plt.tight_layout()

            else: # if no data exists at time point 1
                axs.scatter(pointsA[:, 0], pointsA[:, 1], marker='o', c='blue', label='Initial Points')
                axs.set_xticks([])
                axs.set_yticks([])
                axs.legend(loc='best')
                fig.suptitle(f'{self.well_name}: Aligned Point Sets', fontsize=14)
                plt.tight_layout()

        elif (fig is not None) and (axs is not None):
            # Check dimensions of provided fig and axs and raise error if needed
            expected_shape = (2, n_cols)
            if axs.ndim == 1 and n_cols == 1:
                axs = axs.reshape(2, 1)
            elif axs.shape != expected_shape:
                raise ValueError(f"axs must have shape {expected_shape} for {n_cols+1} time points. Got {axs.shape}")
            # Proceed with plotting as before if axs size checks out

        else: # create subplots if figures and axes weren't provided.
            fig, axs = plt.subplots(2, n_cols, figsize=(3*n_cols, 6))
            if axs.ndim == 1 and n_cols == 1:
                axs = axs.reshape(2, 1)

        if marker_shape is None:
            marker_shape = ['o', '^', 's', 'd', '*'] 
        if marker_colors is None:
            marker_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if title_legend is None:
            title_legend = [f"Time {i+1}" for i in range(len(self.aligned_points) - 1)] #adjust the title legend

        # Collect legend handles and labels for the entire figure
        legend_handles = []
        legend_labels = []

        # Main plotting loop 
        for j in range(len(self.aligned_points) - 1):
            pointsA = self.aligned_points[0] #always relative to the first timepoint
            pointsB = self.aligned_points[j+1]
            pointsB_unaligned = self.unaligned_points[j] #access unaligned points for plotting.

            # Plot with labels for the combined legend (but not displayed on subplots):
            label0 = title_legend[0] if j == 0 else "" #only add pointsA label for the first time point
            label_unaligned = f'Points B (Time {title_legend[j + 1]})' if j == 0 else "" #only add for first time point
            label_aligned = f'Aligned Points B (Time {title_legend[j+1]})' if j == 0 else ""

            # Plotting logic 
            axs[0, j].scatter(pointsA[:, 0], pointsA[:, 1], marker=marker_shape[0], c=marker_colors[0], label='Points A (Original Polygon)')
            axs[0, j].scatter(pointsB_unaligned[:, 0], pointsB_unaligned[:, 1], marker=marker_shape[j + 1], c=marker_colors[j + 1], label=f'Points B (Time {j + 1})') # plot unaligned points
            axs[0, j].set_xticks([]) # updated from axs[0, j-1].set_xticks([])
            axs[0, j].set_yticks([])  # updated from axs[0, j-1].set_yticks([])
            axs[0, j].set_title(title_legend[j+1], fontsize=10) #adjust title
            axs[0, j].legend(loc='best', **kwargs) #add legend for each subplot

            axs[1, j].scatter(pointsA[:, 0], pointsA[:, 1], marker=marker_shape[0], c=marker_colors[0], s=10, alpha=0.7, label='Points A (Original Polygon)')
            axs[1, j].scatter(pointsB[:, 0], pointsB[:, 1], marker=marker_shape[j+1], c=marker_colors[j+1], label=f'Aligned Points B (Time {j+1})')
            axs[1, j].plot(np.append(pointsA[:, 0], pointsA[0, 0]), np.append(pointsA[:, 1], pointsA[0, 1]), marker_colors[0], linewidth=3.0, alpha=0.6)
            axs[1, j].plot(np.append(pointsB[:, 0], pointsB[0, 0]), np.append(pointsB[:, 1], pointsB[0, 1]), marker_colors[j+1], alpha=1.0, linewidth=1.0)
            axs[1, j].set_xticks([])
            axs[1, j].set_yticks([])
            axs[1, j].legend(loc='best', **kwargs) #add legend for each subplot

        # Get handles and labels *outside* loop ONCE after plotting all time points
        handles, labels = axs[0, 0].get_legend_handles_labels()  # Get from the first subplot
        legend_handles.extend(handles)
        legend_labels.extend(labels)
        
        for jj in range(n_cols): #for each time point after the first one, add the labels for the aligned points. These should all be unique and have unique handles
            handles_aligned, labels_aligned = axs[1, jj].get_legend_handles_labels()
            legend_handles.append(handles_aligned[1]) # index 1 has the aligned points. index 0 is redundant.
            legend_labels.append(labels_aligned[1])

        # adding labels to each of the rows in the figure
        axs[0, 0].set_ylabel('Centered Points', **kwargs)
        axs[1, 0].set_ylabel('Aligned Geometry', **kwargs)

        # Create a single legend for the entire figure
        ncol = kwargs.get('ncol', 1)
        fig.legend(legend_handles, title_legend, loc='lower center', ncol=min(len(legend_labels), 8), bbox_to_anchor=(0.5,-0.1)) 

        # remove the individual subplot legends
        [ax.get_legend().remove() for ax in axs.ravel()]
        
        fig.suptitle(f'{self.well_name}: Aligned Point Sets', **kwargs) #updated suptitle
        plt.tight_layout()
        if save:
            save_dir = save_dir or 'alignment_output'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, f"{self.well_name}_aligned_points.png"))  # Updated filename
        if show_plot:
            plt.show()

    def cluster_labeling(self, seed=0):
        """
        Clusters the aligned points in the registered point space to assign labels to the aligned points.
        """
        # get the registered dataframe after alignment algorithm is run
        registered = self.registered_df.copy()
        aligned_points = registered[['registered_x', 'registered_y']]

        n_clusters = len(registered['original_label'].unique())

        # Apply k-means clustering to ALIGNED points
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed) 
        cluster_labels = kmeans.fit_predict(aligned_points) #fit to aligned
        
        # Add cluster centers to the Aligner2D object
        self.cluster_centers_ = kmeans.cluster_centers_

        # make a map that matches cluster labels to the point label at t=0
        label_tuples = zip(
            cluster_labels[0:n_clusters],
            registered['original_label'].values[0:n_clusters]
        )
        label_map = {t[0]:t[1] for t in label_tuples}

        # get a column of labels
        mapped_labels = [label_map[c] for c in cluster_labels]

        # copy the registered data frame and add cluster labels
        self.cluster_labeled_df = registered
        self.cluster_labeled_df['cluster_labels'] = mapped_labels

        return self
        


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