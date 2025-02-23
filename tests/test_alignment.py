import unittest
import numpy as np
from align_spheroid.alignment import Aligner2D

class TestAligner2D(unittest.TestCase):

    def setUp(self):
        # Example data (replace with your actual test data)
        self.dataX = np.random.rand(2, 4, 5)  # 2 wells, 4 time points, 5 cells/points
        self.dataY = np.random.rand(2, 4, 5)
        self.aligner = Aligner2D(well_name="Test_Well") #create an instance

    def test_align_single_point(self):
        dataX_single = np.random.rand(2, 2, 1) # 2 wells, 2 time points, 1 point
        dataY_single = np.random.rand(2, 2, 1)
        aligner_single = Aligner2D(well_name="Test_Well_Single_Point")
        aligner_single.align(dataX_single, dataY_single, well_id=0, time_points = list("ab"))
        self.assertEqual(len(aligner_single.aligned_points), 2)

    def test_align(self):
        self.aligner.align(self.dataX, self.dataY, well_id=0, time_points = list("abcd"))
        self.assertEqual(len(self.aligner.aligned_points), 4)  # Check if the correct number of time points were aligned
        self.assertEqual(len(self.aligner.aligned_points[0]), 5) #check if the number of points at time point 0 is correct
        self.assertTrue(isinstance(self.aligner.registered_df, pd.DataFrame)) #check if the output dataframe is constructed.

if __name__ == "__main__":
    unittest.main()