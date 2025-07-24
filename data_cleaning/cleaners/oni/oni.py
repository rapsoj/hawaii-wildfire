import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, Any, Union, Annotated
import os
from pathlib import Path
from loguru import logger
import requests
import typer


from base_cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Example cleaner that generates and processes synthetic in-memory data"""

    def get_metadata(self) -> Dict[str, Any]:
        """Cleaner for Oceanic Niño Index (ONI) data"""
        return {
            'source': 'Climate Prediction Center (CPC) ONI',
            'description': 'Three-month running mean of sea surface temperature anomalies in the Niño 3.4 region',
            'update_frequency': 'Monthly (first Thursday of each month with month lag)',
            'url': 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
        }

    def download_data(self, format: str = 'dataframe') -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
        """Generate synthetic data and return it in the requested format"""
        self.logger.info("Generating synthetic data...")

        SOURCE_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"        
        CURRENT_DIR = Path(__file__).parent                   
        DATA_ROOT = CURRENT_DIR.parent.parent / "data"           
        FOLDER_PATH = DATA_ROOT / "teleconnections"
        OUT_FILE = FOLDER_PATH / "oni.txt"

        response = requests.get(SOURCE_URL)
        FOLDER_PATH.mkdir(parents=True, exist_ok=True)

        with OUT_FILE.open("w") as fp:
            fp.write(response.text)

        df = pd.read_table(os.path.join(FOLDER_PATH, "oni.txt"),
                           sep='\s+', engine='python')

        self.logger.info(f"Generated {len(df)} records")

        return df


    def clean_data(self, raw_data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
        """Cleaning data..."""
        cleaned = raw_data.copy()

        cleaned = cleaned.rename(columns={'YR':'year'})
        cleaned.columns = cleaned.columns.str.strip()
        cleaned = cleaned.rename(columns={c: 'oni' + c for c in cleaned.columns if c not in ['year', 'month']})

        month_conversion_dictionary = {
            'DJF': 1,
            'JFM': 2,
            'FMA': 3,
            'MAM': 4,
            'AMJ': 5,
            'MJJ': 6,
            'JJA': 7,
            'JAS': 8,
            'ASO': 9,
            'SON': 10,
            'OND': 11,
            'NDJ': 12
            }
        
        cleaned['month'] = cleaned.oniSEAS.map(month_conversion_dictionary.get)
        
        return cleaned.drop(columns='oniSEAS')