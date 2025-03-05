# synthetic data generation to test spheroid alignment algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import string
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix

class CellSimulator:
    def __init__(self, n_circles=6, width=1000, height=1000, min_radius=35, 
                 max_radius=50, min_distance=50, rng=None):
        """Simulates movement of cells represented as circles.

        Args:
            n_circles (int): Number of cells to simulate.
            width (int): Width of the simulation space.
            height (int): Height of the simulation space.
            min_radius (int): Minimum cell radius.
            max_radius (int): Maximum cell radius.
            min_distance (int): Minimum distance between cells.
            rng (np.random.Generator): Random number generator.
        """
        self.n_circles = n_circles
        self.width = width
        self.height = height
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_distance = min_distance
        self.rng = rng if rng is not None else np.random.default_rng() #added for reproducibility
        self.labels = list(string.ascii_lowercase)[:self.n_circles] #initialize labels
        self.circles_df = self._generate_initial_circles()
        
        if self.circles_df is None:
            raise ValueError("Failed to generate initial cell positions.")
            
    def _generate_initial_circles(self):
         """Generates initial non-overlapping circles using Bridson's algorithm."""
         # Poisson-disc sampling using Bridson's algorithm could ensure an efficient no-overlap. 
         # This is more efficient if you require a more dense arrangement while avoiding overlap.
         # Look into a Bridson algorithm package if you want this level of density.
         # Otherwise, the updated code is sufficient.
         radii = self.rng.integers(self.min_radius, self.max_radius, endpoint=True, size=self.n_circles)
         circles_data = []
         for r in radii:
             attempts = 0
             max_attempts = 1000  # Adjust as needed
             while attempts < max_attempts:
                 x = self.rng.uniform(r + self.min_distance, self.width - r - self.min_distance)
                 y = self.rng.uniform(r + self.min_distance, self.height - r - self.min_distance)
                 if all(np.sqrt((x - c['x'])**2 + (y - c['y'])**2) >= r + c['radius'] + self.min_distance for c in circles_data):
                     circles_data.append({'x': x, 'y': y, 'radius': r})
                     break
                 attempts += 1
             else:
                 # Properly handle failure to place after max_attempts:
                 raise ValueError("Failed to place all cells without overlaps.")

         circles_df = pd.DataFrame(circles_data)
         circles_df.insert(0, 'label', self.labels)  # labels created only once
         return circles_df

    def rotate(self, angle_degrees, center_x=None, center_y=None):
        """Rotates all cells by a given angle."""
        if center_x is None:
            center_x = self.width / 2
        if center_y is None:
            center_y = self.height / 2
        angle_rad = np.deg2rad(angle_degrees)
        self.circles_df[['x', 'y']] = self.circles_df.apply(
            lambda row: pd.Series(rotate_point(row['x'], row['y'], angle_rad, center_x, center_y)), axis=1
        )


    def translate(self, tx=0, ty=0): 
        """Translates all cells by a fixed x and y amount.

        Args:
            tx (int or float, optional): Translation in x direction. Defaults to 0.
            ty (int or float, optional): Translation in y direction. Defaults to 0.
        """

        self.circles_df['x'] += tx 
        self.circles_df['y'] += ty

    def jitter(self, max_jitter_x, max_jitter_y): # jitter method to randomly move cells by random amounts.
        """Jitters (randomly translates) all cells within given bounds."""

        jx = self.rng.integers(-max_jitter_x, max_jitter_x, endpoint=True, size=self.n_circles) 
        jy = self.rng.integers(-max_jitter_y, max_jitter_y, endpoint=True, size=self.n_circles)

        self.circles_df['x'] += jx
        self.circles_df['y'] += jy

    def move_individual_cells(self, movement_specs, mvmnt_type='min', use_nn_percentage=False):
        """Moves specified individual cells.

        Args:
            movement_specs (dict): Dictionary where keys are cell labels and values 
                                   are movement distances (absolute or percentage).
            use_radius_percentage (bool): If True, movement is a percentage of radius, 
                                        otherwise it's a percentage of distance to next cell.
        """

        for label, movement_percentage in movement_specs.items():
            try:
                row_index = self.circles_df.index[self.circles_df['label'] == label][0] #get row index
            except IndexError:
                print(f"Warning: Cell with label '{label}' not found, skipping.")
                continue
            
            if mvmnt_type == 'radius':
                radius = self.circles_df.loc[row_index, 'radius']
                movement_amount = movement_percentage * radius
            elif mvmnt_type == 'nearest':
                # Calculate movement based on distance to the next cell
                current_index = self.circles_df.index.get_loc(self.circles_df[self.circles_df.label == label].index[0])
                next_index = (current_index + 1) % len(self.circles_df)
                next_cell = self.circles_df.iloc[next_index]
                distance_to_next = np.sqrt((self.circles_df.loc[row_index, 'x'] - next_cell['x'])**2 + (self.circles_df.loc[row_index, 'y'] - next_cell['y'])**2)
                movement_amount = movement_percentage * distance_to_next
            elif mvmnt_type == 'min':
                # calculate movement as a percentage of the minimum distance between cells at initial
                path = self.circles_df[['x','y']].to_numpy()
                distances = distance_matrix(path, path)
                # Exclude distances to self (diagonal elements)
                distances[np.diag_indices_from(distances)] = np.inf
                movement_amount = movement_percentage * np.min(distances)

            # Apply random direction
            angle = self.rng.uniform(0, 2 * np.pi) #random direction
            dx = movement_amount * np.cos(angle) #x movement
            dy = movement_amount * np.sin(angle) #y movement
            self.circles_df.loc[row_index, 'x'] += dx
            self.circles_df.loc[row_index, 'y'] += dy

    def _permute_rows(self, df):
        """ Shuffles the rows in a pandas dataframe. """
        df_shuffled = df.sample(frac=1).reset_index(drop=True) 
        return df_shuffled

    def simulate(
        self, 
        time_points, 
        rotations=None, 
        translations=None, 
        jitters=None,
        individual_movements=None, 
        mvmnt_type='min',
        shuffle_points=True
    ):
        """Simulates cell movement over multiple time points.


        Args:
            time_points (list): List of time point labels (e.g., ['0h', '24h']).
            rotations (list): List of rotation angles in degrees for each time point (after the first).
            translations (list): List of (max_translate_x, max_translate_y) tuples for each time point (after the first).
            individual_movements (list):  List of dictionaries specifying individual cell movements for each time point (after the first).
            use_radius_percentage (bool): If True, movement is a percentage of radius, 
                                        otherwise it's a percentage of distance to next cell.

        Returns:
            pandas.DataFrame: DataFrame containing cell positions at each time point.

        """

        if rotations is not None and len(rotations) != len(time_points) -1:
            raise ValueError("Number of rotations must be one less than the number of time points.")
        if translations is not None and len(translations) != len(time_points) -1:
            raise ValueError("Number of translations must be one less than the number of time points.")
        if individual_movements is not None and len(individual_movements) != len(time_points) -1:
            raise ValueError("Number of individual_movements must be one less than the number of time points.")

        # Initialize results DataFrame
        all_data = [pd.DataFrame({'label': self.circles_df['label'].tolist(), f'x_{time_points[0]}': self.circles_df['x'], f'y_{time_points[0]}': self.circles_df['y']})] #added label for use in downstream matching, since indices might get shuffled
        
        for i, time in enumerate(time_points[1:]):
            # Key Change: Call rotate and translate on the simulator object
            if rotations is not None:
                self.rotate(rotations[i])  # Use self.rotate

            if translations is not None:
                self.translate(*translations[i])  # Use self.translate

            if individual_movements is not None: #check if this parameter was provided
                self.move_individual_cells(individual_movements[i], mvmnt_type=mvmnt_type)

            if jitters is not None: #apply jitter
                self.jitter(*jitters[i]) #use the jitter method

            tp_df = pd.DataFrame({
                        f'label_{time}': self.circles_df['label'],
                        f'x_{time}': self.circles_df['x'],
                        f'y_{time}': self.circles_df['y']
                    })
            
            # shuffle the rows of the data analogous to recording data points out of order
            if shuffle_points:
                tp_df = self._permute_rows(tp_df)
                
            all_data.append(tp_df)

            #all_data[f'x_{time}'] = self.circles_df['x'] #access from self.circles_df since rotate modifies this in place
            #all_data[f'y_{time}'] = self.circles_df['y']
            #all_data[f'label_{time}'] = self.circles_df['label']

        simulated_data = pd.concat(all_data, axis=1)
            
        self.simulated_data = simulated_data
        return simulated_data

##########

def generate_initial_circles(
    n = 6,
    width = 500, 
    height = 500, 
    min_radius = 35, 
    max_radius = 50, 
    min_distance = 50, 
    rng = None,
    **kwargs):
    """Generates initial non-overlapping circles."""
    if rng is None:
        seed = kwargs.get('seed', 123)
        rng = np.random.default_rng(seed)
    space = np.zeros((height, width))
    circles_data = []
    for _ in range(n):
        radius = rng.integers(min_radius, max_radius, endpoint=True)
        attempts = 0
        max_attempts = kwargs.get('max_attempts', 1000)
        while attempts < max_attempts:
            x = rng.integers(max_radius + min_distance, width - max_radius - min_distance, endpoint=True)
            y = rng.integers(max_radius + min_distance, height - max_radius - min_distance, endpoint=True)
            overlap = False
            for existing_circle in circles_data:
                dist = np.sqrt((x - existing_circle['x'])**2 + (y - existing_circle['y'])**2)
                if dist < radius + existing_circle['radius'] + min_distance:
                    overlap = True
                    break
            if not overlap:
                break
            attempts += 1
        if attempts == max_attempts:
            return None, None  # Handle failure
        #add each circle to the space array for plotting purposes
        for i in range(height):
            for j in range(width):
                if (i - y)**2 + (j - x)**2 <= radius**2:
                    space[i,j] = 1
        circles_data.append({'x': x, 'y': y, 'radius': radius})
    circles_df = pd.DataFrame(circles_data)
    # Assign labels (you can modify the labeling scheme if needed)
    alphabet = list(string.ascii_lowercase)
    circles_df.insert(0, 'label', alphabet[:circles_df.shape[0]])
    return space, circles_df

def rotate_point(x, y, angle_rad, center_x, center_y):
    """Rotates a point around a center."""
    x -= center_x
    y -= center_y
    rotated_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    rotated_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return rotated_x + center_x, rotated_y + center_y

def translate_point(x, y, max_translate_x, max_translate_y, rng):
    """Translates a point randomly within a threshold."""
    tx = rng.integers(-max_translate_x, max_translate_x, endpoint=True)
    ty = rng.integers(-max_translate_y, max_translate_y, endpoint=True)
    return x + tx, y + ty


def assign_nearest_labels(initial_df, current_df):
    """Assigns labels based on nearest initial point."""

    dist_matrix = cdist(initial_df[['x', 'y']], current_df[['x', 'y']])
    nearest_indices = np.argmin(dist_matrix, axis=0)

    # Ensure unique label assignments
    assigned_labels = {}
    current_labels = []
    for i in range(len(current_df)):
        nearest_label = initial_df.iloc[nearest_indices[i]]['label']
        while nearest_label in assigned_labels:  # Check for duplicates
            nearest_indices[i] = (nearest_indices[i] + 1) % len(initial_df) # pick the next closest
            nearest_label = initial_df.iloc[nearest_indices[i]]['label'] # assign that label
        assigned_labels[nearest_label] = True  # Mark label as assigned
        current_labels.append(nearest_label)
    current_df['label_false'] = current_labels
    current_df = current_df.sort_values(by='label_false')

    return current_df

def permute_rows(df):
    """ Shuffles the rows in a pandas dataframe. """
    df_shuffled = df.sample(frac=1).reset_index(drop=True) 
    return df_shuffled

def simulate_circles(
    n_circles = 6, 
    width = 1000, 
    height = 1000, 
    min_radius = 35, 
    max_radius = 50, 
    min_distance = 50, 
    max_translate = 30, 
    time_points = ["0h", "24h", "48h", "72h", "168h"], 
    initial_seed = 123,
    **kwargs
):
    """Simulates circle movement over time."""

    rng_seeds = np.random.default_rng(initial_seed) #use initial seed to create base rng
    seeds = rng_seeds.integers(0, 1000000, size=len(time_points)) #generate list of seeds for each time point.

    space, initial_df = generate_initial_circles(
        n_circles, 
        width, 
        height, 
        min_radius, 
        max_radius, 
        min_distance, 
        rng = np.random.default_rng(seeds[0])
    )

    if initial_df is None:
        print("Could not place circles and initial conditions are None")
        return None #handle error

    # Store data for all time points
    all_data = {'label_true': initial_df['label'].tolist()}
    all_data[f'label_{time_points[0]}_true'] = initial_df['label']
    all_data[f'x_{time_points[0]}'] = initial_df['x']
    all_data[f'y_{time_points[0]}'] = initial_df['y']

    # Simulate movement for each time point
    rot_angles = []
    #current_df = initial_df.copy()
    for i, time in enumerate(time_points[1:]):
        current_df = initial_df.copy()
        rng = np.random.default_rng(seeds[i+1])  # Seed RNG for each time point
        angle = rng.uniform(0, 360)  # Random rotation angle
        rot_angles.append(angle)
        center_y, center_x = height // 2, width // 2

        rotated_circles_data = []
        for _, row in current_df.iterrows():
            angle_rad = np.deg2rad(angle)
            rotated_x, rotated_y = rotate_point(row['x'], row['y'], angle_rad, center_x, center_y)
            if max_translate == 0:
                # skip the translation step if translation parameter is 0
                rotated_circles_data.append({'label': row['label'], 'x': rotated_x, 'y': rotated_y, 'radius': row['radius']})  # Correctly includes 'label'
            else:
                translated_x, translated_y = translate_point(rotated_x, rotated_y, max_translate, max_translate, rng)
                rotated_circles_data.append({'label': row['label'], 'x': translated_x, 'y': translated_y, 'radius': row['radius']})  # Correctly includes 'label'

        current_df = pd.DataFrame(rotated_circles_data)
        #current_df['cluster_labels'] = cluster_num #add cluster number
        #current_df = current_df.reset_index(drop = True)

        #current_df = assign_nearest_labels(initial_df, current_df)  # Correctly pass current_df
        current_df['label_false'] = alphabet = list(string.ascii_lowercase)[:current_df.shape[0]]

        # permute the rows to put things out of order
        current_df = permute_rows(current_df)

        all_data[f'label_{time}_true'] = current_df['label_false']
        all_data[f'x_{time}'] = current_df['x']
        all_data[f'y_{time}'] = current_df['y']
    
    all_df = pd.DataFrame(all_data)
    return all_df, rot_angles