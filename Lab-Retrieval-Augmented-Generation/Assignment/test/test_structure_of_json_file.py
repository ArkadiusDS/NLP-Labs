import unittest
import json
import os
import sys
import argparse


class TestResultsStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(cls.json_path, 'r', encoding='utf-8') as f:
            try:
                cls.data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON file: {e}")
                sys.exit(1)

    def test_main_keys_types(self):
        data = self.data
        self.assertIsInstance(data.get("topic"), str)
        self.assertIsInstance(data.get("question"), str)
        self.assertIsInstance(data.get("answer"), str)
        self.assertIsInstance(data.get("rag"), list)

    def test_rag_structure(self):
        for item in self.data["rag"]:
            self.assertIsInstance(item.get("question"), str)
            self.assertIsInstance(item.get("reason"), str)
            self.assertIsInstance(item.get("experiments"), list)

            for exp in item["experiments"]:
                self.assertIsInstance(exp.get("chunk_size"), int)
                self.assertIsInstance(exp.get("chunk_overlap"), int)
                self.assertIsInstance(exp.get("answer"), str)
                self.assertIsInstance(exp.get("reflection"), str)


if __name__ == "__main__":
    # Parse the JSON file argument before unittest sees it
    parser = argparse.ArgumentParser(description="Run structure tests on a JSON file.")
    parser.add_argument("json_path", help="Path to the JSON file to validate.")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"File not found: {args.json_path}")
        sys.exit(1)

    # Attach json_path to the test class
    TestResultsStructure.json_path = args.json_path

    # Clear sys.argv so unittest doesn't try to use it
    sys.argv = [sys.argv[0]]

    unittest.main()
