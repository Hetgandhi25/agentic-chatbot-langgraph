"""
Unit tests for search tool.
"""

import unittest
from tools.search import search_tool


class TestSearchTool(unittest.TestCase):
    def test_search_tool_attributes(self):
        self.assertEqual(search_tool.name, "tavily_search")
        self.assertEqual(search_tool.max_results, 5)


if __name__ == "__main__":
    unittest.main()

