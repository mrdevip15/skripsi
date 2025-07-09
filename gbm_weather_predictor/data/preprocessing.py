"""
Data preprocessing module for GBM Weather Predictor
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from ..config.settings import WIND_DIR_MAP

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scalers = {}
        
    def preprocess_data(self, df):
        """Enhanced preprocessing with interpolation for special values"""
        try:
            df = df.copy()
            
            # Log initial size
            initial_size = len(df)
            print(f"Initial dataset size: {initial_size}")
            
            # Convert date
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
            
            # Handle special values (8888 and 9999) with enhanced interpolation
            special_values = [8888, 9999]
            
            # Identify all columns with special values
            columns_to_interpolate = []
            for col in df.columns:
                if df[col].dtype in [np.int64, np.float64]:
                    if any(df[col].isin(special_values)):
                        columns_to_interpolate.append(col)
            
            if columns_to_interpolate:
                print(f"\nFound special values (8888, 9999) in columns: {', '.join(columns_to_interpolate)}")
                
                # Sort by date for proper interpolation
                df = df.sort_values('Tanggal')
                
                # Process each column with special values using enhanced interpolation
                for col in columns_to_interpolate:
                    # Count special values before replacement
                    special_count = sum(df[col].isin(special_values))
                    
                    if special_count > 0:
                        print(f"Column '{col}': {special_count} special values found")
                        
                        # Create a mask for special values
                        mask = df[col].isin(special_values)
                        
                        # Store original values for reference
                        original_special_positions = df.index[mask].tolist()
                        
                        # Temporarily set special values to NaN for interpolation
                        df.loc[mask, col] = np.nan
                        
                        # Step 1: Forward fill (propagate last valid observation forward)
                        df[col] = df[col].fillna(method='ffill')
                        filled_by_ffill = df.loc[original_special_positions, col].notna().sum()
                        
                        # Step 2: Backward fill (propagate next valid observation backward)
                        df[col] = df[col].fillna(method='bfill')
                        filled_by_bfill = df.loc[original_special_positions, col].notna().sum() - filled_by_ffill
                        
                        # Step 3: For any remaining NaN values, use 5-day rolling mean
                        remaining_nan_mask = df[col].isna()
                        if remaining_nan_mask.any():
                            # Calculate 5-day rolling mean (centered window when possible)
                            rolling_mean_5d = df[col].rolling(window=5, center=True, min_periods=1).mean()
                            
                            # Fill remaining NaN values with 5-day rolling mean
                            df.loc[remaining_nan_mask, col] = rolling_mean_5d.loc[remaining_nan_mask]
                            filled_by_rolling = remaining_nan_mask.sum()
                        else:
                            filled_by_rolling = 0
                        
                        # Final fallback: if any NaNs still remain, use column median
                        final_nan_mask = df[col].isna()
                        if final_nan_mask.any():
                            column_median = df[col].median()
                            df.loc[final_nan_mask, col] = column_median
                            filled_by_median = final_nan_mask.sum()
                            print(f"  - Used median ({column_median:.2f}) for {filled_by_median} remaining values")
                        
                        print(f"  - Forward fill: {filled_by_ffill} values")
                        print(f"  - Backward fill: {filled_by_bfill} values") 
                        print(f"  - 5-day rolling mean: {filled_by_rolling} values")
                        print(f"  - Successfully interpolated all {special_count} special values in '{col}'")
            
            # Convert wind direction to numeric
            # Handle wind direction in ddd_car column - with better error handling
            if 'ddd_car' in df.columns:
                # First clean up the strings (remove trailing spaces)
                if df['ddd_car'].dtype == object:  # Check if it's a string column
                    df['ddd_car'] = df['ddd_car'].astype(str).str.strip()
                    
                # Convert to degrees using mapping, but keep original for prediction
                df['ddd_car_numeric'] = df['ddd_car'].apply(
                    lambda x: WIND_DIR_MAP.get(str(x).strip(), np.nan) if pd.notna(x) else np.nan
                )
                
                # For prediction purposes, we'll use the numeric version
                # But we need to handle the original ddd_car values properly
                # Convert ddd_car to numeric if it contains degree values
                df['ddd_car'] = pd.to_numeric(df['ddd_car'], errors='coerce')
                # Fill NaN values with the mapped numeric values
                df['ddd_car'] = df['ddd_car'].fillna(df['ddd_car_numeric'])
                
                # Handle ddd_x column - ensure it's numeric first
                if 'ddd_x' in df.columns:
                    # Try to convert to numeric, coercing errors to NaN
                    df['ddd_x'] = pd.to_numeric(df['ddd_x'], errors='coerce')
                    # Fill NaN values in ddd_x with values from ddd_car
                    df.loc[df['ddd_x'].isna(), 'ddd_x'] = df.loc[df['ddd_x'].isna(), 'ddd_car']
            
            # Handle missing values in other columns (excluding special values already handled)
            for col in df.columns:
                if df[col].dtype != 'datetime64[ns]' and df[col].isna().any():
                    # Handle numeric columns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Use similar interpolation strategy as for special values
                        print(f"Handling missing values in column '{col}': {df[col].isna().sum()} NaN values")
                        
                        # Forward fill
                        df[col] = df[col].fillna(method='ffill')
                        # Backward fill
                        df[col] = df[col].fillna(method='bfill')
                        
                        # If any NaNs remain, use 5-day rolling mean
                        if df[col].isna().any():
                            rolling_mean_5d = df[col].rolling(window=5, center=True, min_periods=1).mean()
                            df[col] = df[col].fillna(rolling_mean_5d)
                        
                        # Final fallback: use median
                        if df[col].isna().any():
                            median_val = df[col].median()
                            df[col] = df[col].fillna(median_val)
                            
                    # Handle string/object columns
                    elif df[col].dtype == object:
                        # For string columns, use mode (most common value)
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                        df[col] = df[col].fillna(mode_val)
            
            # Final check: drop any rows that still have NaN values (should be very few if any)
            df_before_drop = df.copy()
            df = df.dropna()
            
            # Print any rows that were dropped due to NaN values
            if len(df_before_drop) > len(df):
                print(f"\nRows dropped due to remaining NaN values: {len(df_before_drop) - len(df)}")
                print("Note: These should be minimal after comprehensive interpolation")
            
            print(f"\nFinal dataset size: {len(df)}")
            print(f"Total rows removed: {initial_size - len(df)}")
            
            # Check no NaN values remain
            nan_counts = df.isna().sum()
            if nan_counts.sum() > 0:
                print("\nWARNING: NaN values still exist in the dataset:")
                print(nan_counts[nan_counts > 0])
            else:
                print("\nNo NaN values remain in the dataset - interpolation successful!")
            
            return df
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise 