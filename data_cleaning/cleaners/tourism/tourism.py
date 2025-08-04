import pandas as pd
import numpy as np
import re
import xarray as xr
from typing import Dict, Any, Union

from base_cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Cleaner for Hawaiian tourism data."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'source': 'Department of Business, Economic Development & Tourism',
            'description': 'Data on the number of visitors, days, expenditure, length of stay, daily census, and per person per day ',
            'update_frequency': 'Monthly'
        }

    def download_data(self, format: str = 'dataframe') -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
        self.logger.info("Downloading data...")

        df = pd.read_excel("../data/tourism/arrivals-by-island-1990-through-2022-final.xlsx", 
                          sheet_name="Visitor arrivals by island",
                          header=[1,2])

        self.logger.info(f"Downloaded dataframe with {len(df)} records")

        return df

    def clean_data(self, raw_data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
            self.logger.info("Cleaning data...")

            cleaned = raw_data.copy()

            # Remove last 7 rows (notes)
            cleaned = cleaned.iloc[:-7]

            # Flatten multi-index columns
            cleaned.columns = [f"{col[0]}_{col[1]}".strip('_') if col[1] else col[0] 
                     for col in cleaned.columns]
            
            # Rename first column
            first_col = cleaned.columns[0]
            cleaned = cleaned.rename(columns={first_col: 'Island_or_Month'})

            # Drop fully empty columns
            cleaned = cleaned.dropna(axis=1, how='all')

            # Forward fill 'Island' column
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            cleaned['Island'] = cleaned['Island_or_Month'].where(
                 ~cleaned['Island_or_Month'].isin(months)
                 )
            cleaned['Island'] = cleaned['Island'].ffill()

            cleaned = cleaned[cleaned['Island_or_Month'].isin(months)].copy()
            cleaned.rename(columns={'Island_or_Month': 'Month'}, inplace=True)

            # 'Melt' to long format
            value_cols = [col for col in cleaned.columns if col not in ['Island', 'Month']]

            melted_dfs = []
            for col in value_cols:
                temp_df = cleaned[['Island', 'Month', col]].copy()
    
                # Extract year and arrival type from column name
                if '_' in col:
                    year_part, arrival_type = col.split('_', 1)
                else:
                    year_part = col
                    arrival_type = 'Total'
    
                # Clean the year part - remove asterisks and other non-digit characters
                year_clean = re.sub(r'[^0-9]', '', str(year_part))
    
                # Skip if we can't extract a valid year
                if not year_clean or len(year_clean) != 4:
                    print(f"Skipping column with invalid year: {col}")
                    continue
    
                try:
                    year = int(year_clean)
                    # Sanity check for reasonable year range
                    if year < 1980 or year > 2030:
                        print(f"Skipping column with unreasonable year: {col} (extracted: {year})")
                        continue
            
                    temp_df['Year'] = year
                    temp_df['Arrival_Type'] = arrival_type
                    temp_df['Arrivals'] = temp_df[col]
                    temp_df = temp_df[['Island', 'Month', 'Year', 'Arrival_Type', 'Arrivals']]
        
                    melted_dfs.append(temp_df)
        
                except ValueError as e:
                    print(f"Error processing column {col}: {e}")
                    continue

            cleaned = pd.concat(melted_dfs, ignore_index=True)
    
            # Clean up datatypes and handle missing values
            cleaned['Arrival_Type'] = cleaned['Arrival_Type'].str.strip()
            cleaned['Month'] = cleaned['Month'].str.strip()
    
            # Convert arrivals to numeric, handling any non-numeric values
            cleaned['Arrivals'] = pd.to_numeric(cleaned['Arrivals'], errors='coerce')
    
            # Sort the final dataframe
            cleaned = cleaned.sort_values(by=['Island', 'Year', 'Month', 'Arrival_Type']).reset_index(drop=True)

            # Convert months to numbers
            month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                         'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                         'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            
            cleaned['Month'] = cleaned['Month'].map(month_map)

            self.logger.info(f"Cleaned dataframe with {len(cleaned)} rows")

            return cleaned