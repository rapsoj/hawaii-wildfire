import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, Any
from datetime import datetime
import logging

from base_cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Cleaner for wildfire boundaries data."""

    def get_metadata(self) -> Dict[str, Any]:  
        return {  
        'source_name': 'Pacific Island Wildfire Data', 
        'variables': {
        'year_month': 'Month of fire event in format YYYY-MM',
        'area_ha': 'Area of fire extent in hectares',
        'geometry': 'Vector polygon of fire extent'},  
        'temporal_resolution': 'monthly',  
        'spatial_resolution': 'vector',
        'update_frequency': 'static'
        }

    def download_to_df(self) -> pd.DataFrame:
        """Download data from source if it is dynamic or from the repo if it is static"""
        self.logger.info("Downloading data to memory...")

        # Read fire data
        df = gpd.read_file('../data/fires/fires_1999_2022.shp')

        self.logger.info(f"Downloaded {len(df)} records")
        return df

    def clean_from_df(self, data) -> object:
        """Clean the input tabular data (e.g., DataFrame, GeoDataFrame, etc.)."""
        self.logger.info("Cleaning data...")

        # Make a copy to avoid modifying the original
        df = data.copy()

        # Standardise data types
        df['Year'] = df['Year'].astype(int)
        df['Month'] = df['Month'].astype(int)

        # Fix column names
        df.columns = [col.lower() for col in df.columns]

        # Create column for year-month
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        df['year_month'] = pd.to_datetime(df['year_month'], format='%Y-%m')

        # Select columns of interest
        df = df[['year_month', 'area_ha', 'geometry']]

        self.logger.info(f"Cleaned data has {len(df)} records")
        return df

    def validate_output(self, df: pd.DataFrame) -> bool:
        """Custom validation for the data"""
        if not super().validate_output(df):
            return False

        # Check that we have the expected join key columns
        expected_columns = ['year_month', 'geometry']
        if not all(col in df.columns for col in expected_columns):
            self.logger.error("Missing expected columns")
            return False

        return True