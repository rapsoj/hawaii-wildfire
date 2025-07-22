"""
Spatial array tests that run on spatial datasets
Each test function should return either:
- bool: Simple pass/fail
- dict: {'passed': bool, 'message': str, 'details': dict}
"""

import numpy as np
from typing import Dict, Any

try:
    import xarray as xr
except ImportError:
    xr = None


def test_not_empty(data) -> Dict[str, Any]:
    """Ensure the spatial array is not empty"""
    if isinstance(data, np.ndarray):
        size = data.size
    elif xr and isinstance(data, (xr.DataArray, xr.Dataset)):
        size = data.size
    else:
        return {'passed': False, 'message': 'Unsupported data type', 'details': {}}

    passed = size > 0
    return {
        'passed': passed,
        'message': f"Data has {size} values" if passed else "Data is empty",
        'details': {'total_elements': int(size)}
    }


def test_has_crs(data=None, cleaner=None) -> Dict[str, Any]:
    """Check that CRS (coordinate reference system) is defined either in cleaner metadata or in data."""

    crs = None

    if cleaner is not None:
        metadata = getattr(cleaner, "get_metadata", lambda: {})()
        print("Cleaner metadata:", metadata)  # Debug print
        crs = metadata.get("CRS")

    # Only check data if no CRS found in metadata
    if crs is None and data is not None and xr and isinstance(data, xr.DataArray):
        crs = getattr(getattr(data, "rio", None), "crs", None)

    passed = crs is not None
    return {
        'passed': passed,
        'message': f"CRS: {crs}" if passed else "No CRS found in cleaner metadata or data",
        'details': {'crs': str(crs) if crs else None}
    }
    

def test_has_valid_data_values(data) -> Dict[str, Any]:
    """Ensure the data is not entirely NaN (some valid values exist)"""
    if isinstance(data, np.ndarray):
        total = data.size
        valid = np.count_nonzero(~np.isnan(data))
    elif xr and isinstance(data, (xr.DataArray, xr.Dataset)):
        total = data.size
        valid = int(data.notnull().sum().item())
    else:
        return {
            'passed': False,
            'message': 'Unsupported data type',
            'details': {}
        }

    passed = valid > 0
    return {
        'passed': passed,
        'message': f"{valid} valid values out of {total}" if passed else "All values are NaN",
        'details': {
            'total_values': int(total),
            'valid_values': int(valid),
            'nan_values': int(total - valid),
            'nan_percentage': round(100 * (total - valid) / total, 2) if total > 0 else None
        }
    }
    

def test_dimensions_valid(data) -> Dict[str, Any]:
    """Ensure spatial dimensions are non-zero"""
    if isinstance(data, np.ndarray):
        shape = data.shape
    elif xr and isinstance(data, (xr.DataArray, xr.Dataset)):
        shape = data.shape
    else:
        return {'passed': False, 'message': 'Unsupported data type', 'details': {}}

    passed = all(dim > 0 for dim in shape)
    return {
        'passed': passed,
        'message': f"Shape is {shape}" if passed else "Invalid dimension size(s)",
        'details': {'shape': shape}
    }


def test_memory_usage_reasonable(data) -> Dict[str, Any]:
    """Ensure the dataset is not too large"""
    max_memory_bytes = 6_000_000_000  # 1 GB

    if isinstance(data, np.ndarray):
        memory_bytes = data.nbytes
    elif xr and isinstance(data, (xr.DataArray, xr.Dataset)):
        memory_bytes = data.nbytes
    else:
        return {'passed': False, 'message': 'Unsupported data type', 'details': {}}

    passed = memory_bytes < max_memory_bytes
    return {
        'passed': passed,
        'message': f"{memory_bytes / (1024**2):.2f} MB used",
        'details': {
            'memory_mb': round(memory_bytes / (1024**2), 2),
            'threshold_mb': round(max_memory_bytes / (1024**2), 2)
        }
    }


def test_value_range_sane(data) -> Dict[str, Any]:
    """Check that values fall within a plausible range"""
    min_val, max_val = None, None

    if isinstance(data, np.ndarray):
        if data.size == 0:
            return {'passed': False, 'message': 'Array is empty', 'details': {}}
        finite_vals = data[np.isfinite(data)]
        if finite_vals.size == 0:
            return {'passed': False, 'message': 'All values are non-finite', 'details': {}}
        min_val, max_val = finite_vals.min(), finite_vals.max()
    elif xr and isinstance(data, xr.DataArray):
        finite_vals = data.where(np.isfinite(data), drop=True)
        if finite_vals.size == 0:
            return {'passed': False, 'message': 'All values are non-finite', 'details': {}}
        min_val, max_val = float(finite_vals.min().values), float(finite_vals.max().values)
    else:
        return {'passed': False, 'message': 'Unsupported data type', 'details': {}}

    valid_range = (-1e6, 1e6)  # Adjust as needed for your domain
    passed = valid_range[0] <= min_val <= max_val <= valid_range[1]

    return {
        'passed': passed,
        'message': f"Values range from {min_val:.2f} to {max_val:.2f}",
        'details': {
            'min': min_val,
            'max': max_val,
            'valid_range': valid_range
        }
    }
