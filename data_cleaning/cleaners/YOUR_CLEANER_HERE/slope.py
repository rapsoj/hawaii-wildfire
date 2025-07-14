import pandas as pd
from pathlib import Path
from typing import Dict, Any
import sys
import numpy as np
import rasterio
import rioxarray

# Add parent directories to path to import base_cleaner
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Cleaner for Hawaii slope data."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'source': 'LANDFIRE Slope Data',
            'description': 'Raster slope data for Hawaii',
            'update_frequency': 'None',
            'CRS': 'ESRI:102007',
            'url': 'https://landfire.gov/node/5684'
        }

    def download_to_df(self) -> pd.DataFrame:
        self.logger.info("Downloading data to memory...")
        # read elevation data
        raster_slope = rioxarray.open_rasterio("../data/elevation/LF2020_SlpD_220_HI/Tif/LH20_SlpD_220.tif")
        # convert to dataframe
        df = raster_slope.to_dataframe(name='slope').reset_index()

        self.logger.info(f"Downloaded {len(df)} records")
        return df


    def clean_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Cleaning data...")
        
        cleaned_df = df.copy()

        # remove redundant columns (band (all 1) and spatial ref (all 0))
        cleaned_df = cleaned_df.drop(['band', 'spatial_ref'], axis = 1)

        # replace nodata value (32767) with NaN
        cleaned_df['slope'] = cleaned_df['slope'].replace(32767, np.nan)

        # convert -9999 elevation values (representing ocean areas) to NaN
        cleaned_df['slope'] = cleaned_df['slope'].replace(-9999, np.nan)

        # drop NaN values
        cleaned_df = cleaned_df.dropna(subset=['slope'])

        self.logger.info(f"Cleaned data has {len(df)} records")

        return cleaned_df
    