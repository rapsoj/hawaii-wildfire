import pandas as pd
import numpy as np
import h5py
import rioxarray
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import tempfile
import gc
from datetime import datetime

from base_cleaner import BaseCleaner


class Cleaner(BaseCleaner):
    """Cleaner for historical rainfall raster data with incremental updates."""

    def __init__(self):
        super().__init__()
        self.hdf5_path = Path("../data/rainfall/cleaned/rainfall_cube.h5")
        
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'source': 'Hawaii Climate Data Portal',
            'description': 'High-resolution gridded rainfall data, updated in near-real-time',
            'update_frequency': 'Monthly',
            'CRS': 'EPSG:4326',
            'url': 'https://www.hawaii.edu/climate-data-portal/data-portal/'
        }

    def _get_existing_times(self) -> List[pd.Timestamp]:
        """Get list of times already processed in the HDF5 file."""
        if not self.hdf5_path.exists():
            return []
        
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'time' not in f:
                    return []
                time_strings = [t.decode('utf-8') for t in f['time'][:]]
                return [pd.to_datetime(t) for t in time_strings]
        except Exception as e:
            self.logger.warning(f"Could not read existing times: {e}")
            return []

    def _scan_for_new_files(self, root_dir: Path, start_year: int = 1990, 
                           end_year: Optional[int] = None) -> List[tuple]:
        """Scan directory for new files that haven't been processed yet."""
        if end_year is None:
            end_year = datetime.now().year
            
        existing_times = set(self._get_existing_times())
        self.logger.info(f"Found {len(existing_times)} existing time periods")
        
        new_files = []
        all_months = [f"{i:02d}" for i in range(1, 13)]
        
        for year in range(start_year, end_year + 1):
            for month in all_months:
                time_period = pd.to_datetime(f"{year}-{month}")
                
                # Skip if we already have this time period
                if time_period in existing_times:
                    continue
                    
                file_path = root_dir / str(year) / f"rainfall_new_month_statewide_data_map_{year}_{month}.tif"
                if file_path.exists():
                    new_files.append((file_path, year, month, time_period))
                    
        self.logger.info(f"Found {len(new_files)} new files to process")
        return sorted(new_files, key=lambda x: x[3])  # Sort by time

    def download_data(self, format: str = 'dataframe') -> Union[pd.DataFrame, np.ndarray, xr.DataArray]:
        """Download new rainfall data files that haven't been processed yet."""
        self.logger.info("Scanning for new rainfall data...")

        root_dir = Path("../data/rainfall/new/month/statewide/data_map")
        
        # For testing, you can limit the years:
        # new_files = self._scan_for_new_files(root_dir, start_year=2024, end_year=2024)
        
        # For full processing:
        new_files = self._scan_for_new_files(root_dir)

        if not new_files:
            self.logger.info("No new files found - skipping cleaning")
            return None

        self.logger.info(f"Processing {len(new_files)} new files...")

        # Process new files
        arrays = []
        time_index = []
        
        for file_path, year, month, time_period in new_files:
            self.logger.info(f"Loading new file: {file_path.name}")
            
            try:
                raster = rioxarray.open_rasterio(file_path).squeeze()
                arrays.append(raster)
                time_index.append(time_period)
                
                # Garbage collection for memory management
                if len(arrays) % 5 == 0:
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue

        if not arrays:
            self.logger.info("No new files successfully loaded - loading existing data")
            return self._load_existing_data()

        # Create DataArray from new files
        self.logger.info(f"Concatenating {len(arrays)} new arrays...")
        new_data = xr.concat(arrays, dim='time')
        new_data['time'] = time_index
        
        # Clear arrays to free memory
        del arrays
        gc.collect()
        
        self.logger.info(f"Created new data array with shape: {new_data.shape}")
        return new_data

    def _load_existing_data(self) -> xr.DataArray:
        """Load existing data from HDF5 file."""
        if not self.hdf5_path.exists():
            raise ValueError("No existing data found and no new files to process")
            
        self.logger.info("Loading existing data from HDF5...")
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                data_values = f['rainfall'][:]
                time_strings = [t.decode('utf-8') for t in f['time'][:]]
                
                # Reconstruct coordinates if available
                coords = {'time': pd.to_datetime(time_strings)}
                if 'x' in f and 'y' in f:
                    coords['x'] = f['x'][:]
                    coords['y'] = f['y'][:]
                
                # Create DataArray
                data_array = xr.DataArray(
                    data_values,
                    coords=coords,
                    dims=['time', 'y', 'x'] if 'x' in coords else ['time', 'dim_1', 'dim_2']
                )
                
            self.logger.info(f"Loaded existing data with shape: {data_array.shape}")
            return data_array
            
        except Exception as e:
            self.logger.error(f"Failed to load existing data: {e}")
            raise

    def clean_data(self, raw_data: xr.DataArray) -> xr.DataArray:
        """Clean new data and append to existing HDF5 file."""
        self.logger.info("Cleaning rainfall data...")
        
        # Clean the new data
        data_values = raw_data.values
        cleaned_values = np.where(data_values < 0, np.nan, data_values)
        
        # Stats
        total_pixels = cleaned_values.size
        valid_pixels = np.count_nonzero(~np.isnan(cleaned_values))
        invalid_pixels = total_pixels - valid_pixels
        
        self.logger.info(f"Cleaning complete:")
        self.logger.info(f"  Total pixels: {total_pixels:,}")
        self.logger.info(f"  Valid pixels: {valid_pixels:,} ({100*valid_pixels/total_pixels:.1f}%)")
        self.logger.info(f"  Invalid/NaN pixels: {invalid_pixels:,} ({100*invalid_pixels/total_pixels:.1f}%)")
        
        # Create cleaned DataArray
        cleaned_data = raw_data.copy()
        cleaned_data.values = cleaned_values
        
        # Append to existing HDF5 file (or create new one)
        self._append_to_hdf5(cleaned_data)
        
        # Return the full dataset (existing + new)
        return self._load_existing_data()

    def _append_to_hdf5(self, new_data: xr.DataArray):
        """Append new data to existing HDF5 file."""
        # Ensure output directory exists
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.hdf5_path.exists():
            # Create new file
            self.logger.info(f"Creating new HDF5 file: {self.hdf5_path}")
            self._create_hdf5_file(new_data)
        else:
            # Append to existing file
            self.logger.info(f"Appending to existing HDF5 file: {self.hdf5_path}")
            self._append_to_existing_hdf5(new_data)

    def _create_hdf5_file(self, data: xr.DataArray):
        """Create a new HDF5 file with the data."""
        with h5py.File(self.hdf5_path, 'w') as f:
            # Save main data with compression and chunking
            f.create_dataset(
                "rainfall", 
                data=data.values, 
                compression="gzip",
                compression_opts=6,
                chunks=True,
                maxshape=(None, data.shape[1], data.shape[2])  # Allow time dimension to grow
            )
            
            # Save time coordinates
            time_strings = [str(t.values) for t in data.time]
            f.create_dataset(
                "time", 
                data=[t.encode('utf-8') for t in time_strings],
                compression="gzip",
                maxshape=(None,)  # Allow to grow
            )
            
            # Save spatial coordinates if available
            if hasattr(data, 'x') and hasattr(data, 'y'):
                f.create_dataset("x", data=data.x.values, compression="gzip")
                f.create_dataset("y", data=data.y.values, compression="gzip")
            
            # Save metadata
            f.attrs['shape'] = data.shape
            f.attrs['last_updated'] = datetime.now().isoformat()
            
        self.logger.info(f"Created HDF5 file with {data.shape[0]} time periods")

    def _append_to_existing_hdf5(self, new_data: xr.DataArray):
        """Append new data to existing HDF5 file."""
        with h5py.File(self.hdf5_path, 'a') as f:
            # Get current size
            current_size = f['rainfall'].shape[0]
            new_size = current_size + new_data.shape[0]
            
            # Resize datasets
            f['rainfall'].resize((new_size, new_data.shape[1], new_data.shape[2]))
            f['time'].resize((new_size,))
            
            # Append new data
            f['rainfall'][current_size:new_size] = new_data.values
            
            # Append time coordinates
            time_strings = [str(t.values) for t in new_data.time]
            new_time_data = [t.encode('utf-8') for t in time_strings]
            f['time'][current_size:new_size] = new_time_data
            
            # Update metadata
            f.attrs['shape'] = (new_size, new_data.shape[1], new_data.shape[2])
            f.attrs['last_updated'] = datetime.now().isoformat()
            
        self.logger.info(f"Appended {new_data.shape[0]} new time periods (total: {new_size})")

    def get_hdf5_info(self) -> Dict[str, Any]:
        """Get information about the HDF5 file."""
        if not self.hdf5_path.exists():
            return {"exists": False}
            
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                info = {
                    "exists": True,
                    "path": str(self.hdf5_path),
                    "file_size_mb": self.hdf5_path.stat().st_size / 1024**2,
                    "shape": f.attrs.get('shape', 'unknown'),
                    "last_updated": f.attrs.get('last_updated', 'unknown'),
                    "time_periods": f['time'].shape[0] if 'time' in f else 0
                }
                
                if 'time' in f:
                    times = [t.decode('utf-8') for t in f['time'][:]]
                    info["time_range"] = f"{times[0]} to {times[-1]}"
                    
                return info
        except Exception as e:
            return {"exists": True, "error": str(e)}