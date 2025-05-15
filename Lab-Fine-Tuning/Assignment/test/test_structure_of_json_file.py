import json
import os
import unittest
import sys

# --- Handle custom argument early ---
if len(sys.argv) < 2:
    print("Usage: python test_experiments.py your_file.json")
    sys.exit(1)

FILE_PATH = sys.argv[1]
sys.argv = sys.argv[:1]

class TestExperimentFileStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"File '{FILE_PATH}' does not exist.")

        with open(FILE_PATH, "r") as f:
            try:
                cls.data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"File is not a valid JSON: {e}")

    def test_contains_four_experiments(self):
        expected_keys = {"experiment_0", "experiment_1", "experiment_2", "final_prediction"}
        self.assertEqual(set(self.data.keys()), expected_keys, "JSON must contain exactly four experiments.")

    def test_experiment_structure(self):
        for exp_name, exp in self.data.items():
            with self.subTest(experiment=exp_name):
                if exp_name == "final_prediction":
                    self.assertIn("experiment_chosen", exp)
                    self.assertIsInstance(exp["experiment_chosen"], str)
                    self.assertIn(exp["experiment_chosen"], ["experiment_0", "experiment_1", "experiment_2"])
                self.assertIn("model", exp)
                self.assertIn("hyperparameters", exp)
                self.assertIn("f1_score", exp)
                self.assertIn("precision", exp)
                self.assertIn("recall", exp)
                self.assertIn("description", exp)

                self.assertIsInstance(exp["model"], str)
                self.assertIsInstance(exp["description"], str)
                self.assertIsInstance(exp["hyperparameters"], dict)
                self.assertIsInstance(exp["f1_score"], (int, float))
                self.assertIsInstance(exp["precision"], (int, float))
                self.assertIsInstance(exp["recall"], (int, float))

                self.assertGreaterEqual(len(exp["hyperparameters"]), 2)
                for hp_key, hp_value in exp["hyperparameters"].items():
                    self.assertIsInstance(hp_key, str)
                    self.assertTrue(isinstance(hp_value, (str, float)))

    def test_descriptions_are_unique(self):
        descriptions = [exp["description"] for exp in self.data.values()]
        self.assertEqual(len(descriptions), len(set(descriptions)), "Descriptions must be unique for each experiment.")

if __name__ == "__main__":
    unittest.main()
