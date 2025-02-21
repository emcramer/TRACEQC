# synthetic data generation to test spheroid alignment algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import string
from scipy.spatial.distance import cdist

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