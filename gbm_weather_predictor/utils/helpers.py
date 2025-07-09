"""
Utility functions for GBM Weather Predictor
"""

import os
import json
from datetime import datetime

def save_metrics(metrics, filepath):
    """Save metrics to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filepath):
    """Load metrics from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_timestamp():
    """Create a timestamp string for file naming"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def sanitize_filename(filename):
    """Sanitize filename by removing problematic characters"""
    problematic_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in problematic_chars:
        filename = filename.replace(char, '_')
    return filename.lower()

def ensure_directory_exists(directory):
    """Ensure a directory exists, create if it doesn't"""
    os.makedirs(directory, exist_ok=True)
    return directory 