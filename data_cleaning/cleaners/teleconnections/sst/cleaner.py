"""
Example data cleaner that works entirely in memory.
Located in: cleaners/example/cleaner.py
To use: python data_cleaning.py --cleaner-name example
"""

import os
from pathlib import Path
import requests
import pandas as pd
import numpy as np


from typing import Dict, Any, Union

from base_cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Cleaner for SST (teleconnections) data."""
    '''SST = Sea Surface Temperature'''

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'source_name': 'Nino Regions Sea Surface Temperatures (SST) Data',
            'description': 'Monthly sea surface temperature anomalies in Niño regions',
            'update_frequency': 'Month end',
            'type': 'Temporal',
            'temporal_resolution': 'Monthly',
            'spatial_resolution': 'None',
            'variables': {
                'YR': 'Year of the measurement',
                'MON': 'Month of the measurement',
                'NINO1+2': 'SST in the Niño 1+2 region of the Pacific Ocean',
                'NINO3': 'SST in the Niño 3 region of the Pacific Ocean',
                'NINO4': 'SST in the Niño 4 region of the Pacific Ocean',
                'NINO3.4': 'SST in the Niño 3.4 region of the Pacific Ocean',
                'ANOM': 'The anomaly from the long-term average temperature (for the corresponding Niño region)'}}

    def download_data(self, format: str = 'dataframe') -> Union[pd.DataFrame, np.ndarray]:
        """Download SST data and return as DataFrame or NumPy array."""

        SOURCE_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
        FILE_PATH_PARTS = ("teleconnections", "sst.txt")
        DATA_ROOT = Path("../data/")
        FOLDER_PATH = os.path.join(str(DATA_ROOT), 'teleconnections')


        self.logger.info("Downloading SST data...")
        response = requests.get(SOURCE_URL)
        out_file = DATA_ROOT.joinpath(*FILE_PATH_PARTS)
        self.logger.info(f"Output file path is {out_file}")
        if out_file.exists():
            self.logger.info("File exists. Skipping.")
        else:
            out_file.parent.mkdir(exist_ok=True, parents=True)
            with out_file.open("w") as fp:
                fp.write(response.text)
            self.logger.info("Data downloaded to file.")
        self.logger.info("Nino Regions SST download complete.")


        #Load the SST data into a DataFrame

        df = pd.read_csv('../data/teleconnections/sst.txt', delimiter= " ")

        self.logger.info(f"Generated {len(df)} records")

        if format == 'dataframe':
            return df
        elif format == 'array':
            # Only numeric columns for NumPy conversion
            return df[['YR', 'MON', 'NINO1+2', 'NINO3', 'NINO4', 'NINO3.4', 'ANOM']].to_numpy()
        else:
            raise ValueError(f"Unsupported format: {format}. Expected 'dataframe' or 'array'.")

    def clean_data(self, raw_data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Clean either a pandas DataFrame or a NumPy array"""
        if isinstance(raw_data, pd.DataFrame):
            cleaned = raw_data.copy()

            # Standardize date column
            cleaned['date'] = pd.to_datetime(cleaned['date'], errors='coerce')

            # Fill missing numeric values
            cleaned['value'] = cleaned['value'].fillna(cleaned['value'].mean())

            # Normalize categorical column
            cleaned['category'] = cleaned['category'].str.upper()

            # Remove outliers in the value column
            cleaned = cleaned[cleaned['value'].between(-100, 200)]

            self.logger.info(f"Cleaned DataFrame with {len(cleaned)} rows")
            return cleaned

        elif isinstance(raw_data, np.ndarray):
            # Remove rows with NaNs
            cleaned = raw_data[~np.isnan(raw_data).any(axis=1)]

            # Remove numeric outliers
            cleaned = cleaned[(cleaned >= -100).all(axis=1) & (cleaned <= 200).all(axis=1)]

            self.logger.info(f"Cleaned array with {len(cleaned)} rows")
            return cleaned

        else:
            raise TypeError("clean_data supports only pandas DataFrame or numpy ndarray")

    def validate_output(self, df: pd.DataFrame) -> bool:
        """Custom validation for the example DataFrame output"""
        if not super().validate_output(df):
            return False

        expected_columns = ['date', 'value', 'category']
        if not all(col in df.columns for col in expected_columns):
            self.logger.error("Missing expected columns")
            return False

        if not df['category'].str.isupper().all():
            self.logger.error("Not all categories are uppercase")
            return False

        return True