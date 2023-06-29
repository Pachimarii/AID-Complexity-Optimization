# import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# sys.path.append("src")  # Adjust the path accordingly
from src.program import evaluate_performance, train_model_random_forest


class TestProgram(unittest.TestCase):
    # does type checks for parameters of evaluate_performance
    def test_evaluate_performance(self):
        y_true = [1, 0, 1]  # Example 1d array-like y_true
        y_pred = [1, 1, 0]  # Example 1d array-like y_pred
        evaluate_performance(y_true, y_pred)

        y_true = np.array([[1, 0], [0, 1]])  # Example label indicator array y_true
        y_pred = np.array([[1, 0], [1, 0]])  # Example label indicator array y_pred
        evaluate_performance(y_true, y_pred)

        invalid_y_true = 123  # Invalid type for y_true
        invalid_y_pred = "labels"  # Invalid type for y_pred
        with self.assertRaises(TypeError):
            evaluate_performance(invalid_y_true, y_pred)
        with self.assertRaises(ValueError):
            evaluate_performance(y_true, invalid_y_pred)

    def test_train_model_randome_forect(self):
        seed = 42
        feature = [[1, 2, 3], [4, 5, 6]]  # Example feature data
        target = [1, 2]  # Example target data

        # Test with valid inputs
        model = train_model_random_forest(seed, feature, target)
        self.assertIsInstance(model, RandomForestRegressor)

        # Test with invalid feature type
        invalid_feature = "invalid"  # Invalid type for feature
        with self.assertRaises(ValueError):
            train_model_random_forest(seed, invalid_feature, target)

        # Test with invalid target type
        invalid_target = {"a": 1, "b": 2}  # Invalid type for target
        with self.assertRaises(TypeError):
            train_model_random_forest(seed, feature, invalid_target)
