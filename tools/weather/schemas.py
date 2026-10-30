"""
Input schemas for weather tool.
"""

from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    location: str = Field(
        ...,
        description="City or location name, for example: 'Ahmedabad', 'London, UK', or 'New York, US'",
    )

