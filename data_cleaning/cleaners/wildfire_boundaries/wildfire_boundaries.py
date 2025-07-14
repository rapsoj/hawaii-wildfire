import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Union

from base_cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Cleaner for Pacific Island wildfire boundaries data."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'source_name': 'Pacific Island Wildfire Data',
            'variables': {
                'year_month': 'Month of fire event in format YYYY-MM',
                'area_ha': 'Area of fire extent in hectares',
                'geometry': 'Vector polygon of fire extent'
            },
            'temporal_resolution': 'monthly',
            'spatial_resolution': 'vector',
            'update_frequency': 'static'
        }

    def download_data(self, format: str = 'dataframe') -> Union[pd.DataFrame, np.ndarray]:
        """Load fire data and return as DataFrame or NumPy array."""
        self.logger.info("Downloading fire boundary data...")

        # Load shapefile
        df = gpd.read_file('../data/fires/fires_1999_2022.shp')

        self.logger.info(f"Loaded {len(df)} records")

        if format == 'dataframe':
            return df
        elif format == 'array':
            # Only numeric columns for NumPy conversion
            return df[['Year', 'Month', 'area_ha']].to_numpy()
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'dataframe' or 'array'.")

    def clean_data(self, raw_data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Clean wildfire boundary data, supporting both DataFrame and array inputs."""
        if isinstance(raw_data, (pd.DataFrame, gpd.GeoDataFrame)):
            df = raw_data.copy()

            # Standardize and clean
            if 'Year' in df.columns and 'Month' in df.columns:
                df['Year'] = df['Year'].astype(int)
                df['Month'] = df['Month'].astype(int)

            df.columns = [col.lower() for col in df.columns]

            df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            df['year_month'] = pd.to_datetime(df['year_month'], format='%Y-%m', errors='coerce')

            # Select final columns
            df = df[['year_month', 'area_ha', 'geometry']]

            self.logger.info(f"Cleaned GeoDataFrame with {len(df)} records")
            return df

        elif isinstance(raw_data, np.ndarray):
            # Remove invalid rows and standardize shape
            cleaned = raw_data[~np.isnan(raw_data).any(axis=1)]
            cleaned = cleaned[(cleaned[:, 2] >= 0)]  # Area must be non-negative

            self.logger.info(f"Cleaned NumPy array with {len(cleaned)} rows")
            return cleaned

        else:
            raise TypeError("clean_data supports only pandas/GeoDataFrame or numpy ndarray")

    def validate_output(self, df: pd.DataFrame) -> bool:
        """Validate final wildfire data"""
        if not super().validate_output(df):
            return False

        expected_columns = ['year_month', 'geometry']
        if not all(col in df.columns for col in expected_columns):
            self.logger.error("Missing expected columns")
            return False

        if not pd.api.types.is_datetime64_any_dtype(df['year_month']):
            self.logger.error("year_month column is not datetime")
            return False

        return True