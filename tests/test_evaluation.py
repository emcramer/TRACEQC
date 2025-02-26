import unittest
from align_spheroid.evaluation import turning_function, frechet_distance, compare_paths
import numpy as np

class TestEvaluation(unittest.TestCase):
    def test_turning_function(self):
        path = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        tf = turning_function(path)
        self.assertTrue(np.allclose(tf, [np.pi/2, np.pi]))

        # Test with a path that has less than 3 points
        path_short = np.array([[0, 0], [1, 1]])
        tf_short = turning_function(path_short) #expect empty array
        self.assertTrue(tf_short.size == 0)

    def test_frechet_distance(self):
        p = np.array([[0, 0], [1, 1], [2, 0]])
        q = np.array([[0, 0], [1, 0], [2, 0]])
        dist = frechet_distance(p, q)
        self.assertAlmostEqual(dist, 1.0)

        p = np.array([[0, 0], [1, 0], [2, 0]])
        q = np.array([[0, 0], [1, 0]])

        dist_uneven = frechet_distance(p,q)

        self.assertEqual(dist_uneven, 2.0)

        #test empty path
        p_empty = np.array([])
        q_empty = np.array([])
        # this should raise an error
        with self.assertRaises(IndexError):
            frechet_distance(p_empty, q_empty)

    def test_compare_paths(self):
        path1 = np.array([[0, 0], [1, 1], [2, 2]])
        path2 = np.array([[0, 0], [1, 0], [2, 0]])
        distance = compare_paths(path1, path2)

        self.assertGreater(distance, 0) #expect non-zero number
        self.assertFalse(np.isinf(distance)) # the result should be a finite number

        # Test with less than 3 points
        path_short = np.array([[1,0], [0,1]])

        dist_short = compare_paths(path_short, path2)

        self.assertTrue(np.isinf(dist_short)) # the distance should be infinite for too short a path



if __name__ == '__main__':
    unittest.main()