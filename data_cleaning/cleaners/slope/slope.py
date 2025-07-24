import pandas as pd
from typing import Dict, Any, Union
import rioxarray
import xarray as xr
import numpy as np


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
    
    def download_data(self, format: str = 'dataframe') -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
        self.logger.info("Downloading elevation data...")

        # read elevation data
        elevation_xarray = rioxarray.open_rasterio("../data/elevation/LF2020_SlpD_220_HI/Tif/LH20_SlpD_220.tif")
        # squeeze to remove band information (irrelevant with single-band raster)
        elevation_array = elevation_xarray.squeeze()
        # convert to numpy array
        elevation_array = elevation_array.to_numpy()

        self.logger.info(f"Downloaded array with {elevation_xarray.count().item()} pixels")
        return elevation_array

    
    def clean_data(self, raw_data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
        self.logger.info("Cleaning data...")

        # replace values <= -9999 with np.nan
        cleaned_array = np.where(raw_data > -9999, raw_data, np.nan)

        # check how many non-NaN pixels remain
        valid_pixel_count = np.count_nonzero(~np.isnan(cleaned_array))

        self.logger.info(f"Cleaned array with {valid_pixel_count} pixels")
        return cleaned_array