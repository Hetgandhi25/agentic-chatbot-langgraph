"""
Unit tests for weather tool.
"""

import unittest
from unittest.mock import patch, MagicMock
from tools.weather import get_current_weather


class TestWeatherTool(unittest.TestCase):
    @patch("tools.weather.tool.requests.get")
    def test_get_current_weather_success(self, mock_get):
        geo_resp = MagicMock()
        geo_resp.json.return_value = {
            "results": [
                {
                    "name": "London",
                    "latitude": 51.5074,
                    "longitude": -0.1278,
                    "country": "United Kingdom",
                    "admin1": "England",
                }
            ]
        }
        geo_resp.raise_for_status = MagicMock()

        weather_resp = MagicMock()
        weather_resp.json.return_value = {
            "current": {
                "temperature_2m": 15.2,
                "apparent_temperature": 14.5,
                "relative_humidity_2m": 72,
                "pressure_msl": 1012,
                "wind_speed_10m": 12.5,
                "visibility": 10000,
                "weather_code": 1,
            }
        }
        weather_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [geo_resp, weather_resp]

        res = get_current_weather.invoke({"location": "London"})
        self.assertIn("Current weather in London, England, United Kingdom:", res)
        self.assertIn("Mainly clear", res)
        self.assertIn("15.2°C", res)


if __name__ == "__main__":
    unittest.main()

