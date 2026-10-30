"""
Unit tests for stock purchase tool.
"""

import unittest
from unittest.mock import patch
from tools.purchase_stock import purchase_stock


class TestPurchaseStockTool(unittest.TestCase):
    @patch("tools.purchase_stock.tool.interrupt")
    def test_purchase_approved(self, mock_interrupt):
        mock_interrupt.return_value = "yes"
        res = purchase_stock.invoke({"symbol": "AAPL", "quantity": 10})
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["symbol"], "AAPL")
        self.assertEqual(res["quantity"], 10)

    @patch("tools.purchase_stock.tool.interrupt")
    def test_purchase_declined(self, mock_interrupt):
        mock_interrupt.return_value = "no"
        res = purchase_stock.invoke({"symbol": "AAPL", "quantity": 10})
        self.assertEqual(res["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()

