import unittest
import numpy as np
import pandas as pd
from align_spheroid.synthetic_data import CellSimulator

class TestCellSimulator(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.rng = np.random.default_rng(123) #set seed
        self.simulator = CellSimulator(n_circles=5, width=500, height=500, min_radius=10, max_radius=20, min_distance=5, rng=self.rng)

    def test_generate_initial_circles(self):
        """Test initial circle generation."""

        df = self.simulator.circles_df
        self.assertEqual(len(df), 5)  # Check correct number of circles
        self.assertTrue(all(col in df.columns for col in ['label', 'x', 'y', 'radius']))  # Check columns
        #check if any cells are overlapping using all vs. any
        self.assertFalse(any(
            np.sqrt((row1['x'] - row2['x'])**2 + (row1['y'] - row2['y'])**2) < row1['radius'] + row2['radius'] + self.simulator.min_distance
            for i, row1 in df.iterrows()
            for j, row2 in df.iterrows()
            if i!=j) #check if any off-diagonal elements in distance matrix are > radius1 + radius2 + min distance
        )

    def test_rotate(self):
        """Test cell rotation."""
        initial_positions = self.simulator.circles_df[['x', 'y']].values.copy()
        self.simulator.rotate(90) #rotate by 90 degrees
        rotated_positions = self.simulator.circles_df[['x', 'y']].values

        self.assertFalse(np.allclose(initial_positions, rotated_positions))  # Positions should change

    def test_translate(self):
        """Test cell translation."""
        initial_positions = self.simulator.circles_df[['x', 'y']].values.copy()
        self.simulator.translate(10, 10)  # Example translation
        translated_positions = self.simulator.circles_df[['x', 'y']].values
        self.assertFalse(np.allclose(initial_positions, translated_positions))  # Check positions changed
        # Add checks for translation within bounds

    def test_individual_movements(self):
        """Test individual cell movement."""
        initial_positions = self.simulator.circles_df[['x', 'y']].values.copy()
        #move cell 'a' by 5% of radius
        movement_specs = {'a': 0.05} 
        self.simulator.move_individual_cells(movement_specs, use_radius_percentage=True)
        moved_positions = self.simulator.circles_df[['x', 'y']].values
        self.assertFalse(np.allclose(initial_positions, moved_positions)) # Check positions changed
        # Add more test conditions

    def test_simulation(self):
        time_points = ["0h", "24h", "48h"]
        rotations = [30, 60]
        translations = [(10, 10), (20, 20)]
        individual_movements = [{'a': 0.1}, {'b': 0.2}]

        sim_df = self.simulator.simulate(time_points=time_points, rotations=rotations, translations=translations, individual_movements=individual_movements)
        #add assertions checking consistency of dataframes, i.e. shape and presence/absence of NAs
        self.assertTrue(all(f'x_{time}' in sim_df.columns and f'y_{time}' in sim_df.columns for time in time_points)) #check x and y at each timepoint
        self.assertEqual(sim_df.shape[0], 5) #ensure number of rows hasn't changed
        self.assertTrue(all(f'label_{time}' in sim_df.columns for time in time_points))


if __name__ == '__main__':
    unittest.main()