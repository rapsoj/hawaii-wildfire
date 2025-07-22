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
import xarray as xr

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

    def download_data(self, format: str = 'dataframe') -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
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

        df = pd.read_csv('../data/teleconnections/sst.txt', sep='\s+')

        self.logger.info(f"Generated {len(df)} records")

        if format == 'dataframe':
            return df
        elif format == 'array':
            # Only numeric columns for NumPy conversion
            return df[['YR', 'MON', 'NINO1+2', 'NINO3', 'NINO4', 'NINO3.4', 'ANOM']].to_numpy()
        else:
            raise ValueError(f"Unsupported format: {format}. Expected 'dataframe' or 'array'.")

    def clean_data(self, raw_data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
        """Clean pandas DataFrame or NumPy array"""
        if isinstance(raw_data, pd.DataFrame):
            cleaned = raw_data.copy()
            
            # Renaming the year and month columns
            cleaned = cleaned.rename(columns={'YR': 'year', 'MON': 'month'})
            
            def rename_column(col):
                if col in ['year', 'month']:
                    return col
                # Replace special characters with underscores to abide by standard_tests.py
                col = col.replace('+', '_plus_').replace('.', '_')
                # Only add nino prefix if the column doesn't already start with NINO
                return col if col.startswith('NINO') else f'nino_{col}'

            
            # Apply the renaming function to all columns
            cleaned.columns = [rename_column(c) for c in cleaned.columns]
            
            # Make column names lowercase
            cleaned.columns = cleaned.columns.str.lower()

            self.logger.info(f"Cleaned DataFrame with {len(cleaned)} rows")
            return cleaned
        
        elif isinstance(raw_data, np.ndarray):
            # Numpy array so just return a copy - main data will be a dataframe anyways
            self.logger.info(f"Input is NumPy array with shape {raw_data.shape}")
            return raw_data.copy()