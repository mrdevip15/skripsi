import os
import shutil
from datetime import datetime

# Source directory containing the files
source_dir = "data bulanan 2010 sampai 2025"

# Get all xlsx files in the directory
files = [f for f in os.listdir(source_dir) if f.endswith('.xlsx')]

# Sort files by timestamp (they're already sorted in the directory listing)
files.sort()

# Process each file
for i, file in enumerate(files):
    # Calculate the date based on the index (starting from January 2010)
    year = 2010 + (i // 12)
    month = (i % 12) + 1
    
    # Create new filename in format MM-YYYY.xlsx
    new_filename = f"{month:02d}-{year}.xlsx"
    
    # Create year folder if it doesn't exist
    year_folder = os.path.join(source_dir, str(year))
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    
    # Full paths for source and destination
    source_path = os.path.join(source_dir, file)
    dest_path = os.path.join(year_folder, new_filename)
    
    # Move and rename the file
    shutil.move(source_path, dest_path)
    print(f"Moved {file} to {dest_path}")

print("File organization complete!") 