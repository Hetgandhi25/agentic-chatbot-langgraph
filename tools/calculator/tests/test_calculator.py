"""
Unit tests for the calculator tool.
"""

import unittest
from tools.calculator import calculator


class TestCalculatorTool(unittest.TestCase):
    def test_basic_arithmetic(self):
        res = calculator.invoke({"expression": "2 + 2"})
        self.assertEqual(res, "4")

    def test_math_module_function(self):
        res = calculator.invoke({"expression": "math.sqrt(16)"})
        self.assertEqual(res, "4.0")

    def test_syntax_error(self):
        res = calculator.invoke({"expression": "2 ++"})
        self.assertTrue(res.startswith("Calculation error:"))

    def test_restricted_builtins(self):
        res = calculator.invoke({"expression": "__import__('os').system('dir')"})
        self.assertTrue(res.startswith("Calculation error:"))


if __name__ == "__main__":
    unittest.main()

