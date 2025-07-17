import pandas as pd
from pathlib import Path
from typing import Dict, Any
import sys
import rasterio
import rioxarray


from base_cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Cleaner for Hawaii elevation data."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'source': 'LANDFIRE Elevation Data',
            'description': 'Raster elevation data for Hawaii',
            'update_frequency': 'None',
            'CRS': 'ESRI:102007',
            'url': 'https://landfire.gov/node/5684'
        }

    def download_to_df(self) -> pd.DataFrame:
        self.logger.info("Downloading data to memory...")
        
        # read elevation data
        raster_array = rioxarray.open_rasterio("../data/elevation/LF2020_Elev_220_HI/Tif/LH20_Elev_220.tif")

        self.logger.info(f"Downloaded {raster_array.count().item()} records")
        return raster_array


    def clean_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Cleaning data...")
        
        cleaned_array = df.copy()

        # convert nodata values (32767) to NaN
        nodata_value = cleaned_array.rio.nodata
        cleaned_array = cleaned_array.where(cleaned_array != nodata_value)

        # replace -9999 values (representing ocean areas) to NaN
        cleaned_array = cleaned_array.where(cleaned_array != -9999)

        self.logger.info(f"Cleaned data has {cleaned_array.count().item()} records")

        return cleaned_array    
