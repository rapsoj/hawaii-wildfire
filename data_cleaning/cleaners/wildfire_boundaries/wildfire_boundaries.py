import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
import logging

from core.base_cleaner import DataCleaner


class Cleaner(DataCleaner):
    """Example cleaner that works entirely in memory"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def download_to_df(self) -> pd.DataFrame:
        """Download data from source if it is dynamic or from the repo if it is static"""
        self.logger.info("Downloading data to memory...")

        # Read fire data
        df = gpd.read_file('data/fires/fires_1999_2022.shp')

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

        # Select columns of interest
        df = df[['year_month', 'area_ha', 'geometry']]

        self.logger.info(f"Cleaned data has {len(df)} records")
        return df

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