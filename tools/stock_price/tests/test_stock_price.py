"""
Unit tests for stock price tool.
"""

import unittest
from unittest.mock import patch, MagicMock
from tools.stock_price import get_stock_price


class TestStockPriceTool(unittest.TestCase):
    @patch("tools.stock_price.tool.requests.get")
    def test_stock_price_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "150.00",
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        res = get_stock_price.invoke({"symbol": "AAPL"})
        self.assertIn("Global Quote", res)
        self.assertEqual(res["Global Quote"]["01. symbol"], "AAPL")

    @patch("tools.stock_price.tool.settings")
    def test_missing_api_key(self, mock_settings):
        mock_settings.alpha_vantage_api_key = ""
        res = get_stock_price.invoke({"symbol": "AAPL"})
        self.assertIn("error", res)
        self.assertIn("ALPHA_VANTAGE_API_KEY is missing", res["error"])


if __name__ == "__main__":
    unittest.main()

