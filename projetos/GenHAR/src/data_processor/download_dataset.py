import requests
import zipfile
import os
from utils import log

def download_zenodo_datasets(extract_dir):
    # Define paths for the data directories
    standardized_view_dir = os.path.join(extract_dir, 'standardized_view')
    baseline_view_dir = os.path.join(extract_dir, 'baseline_view')

    # Check if the directories already exist
    if os.path.exists(standardized_view_dir) and os.path.exists(baseline_view_dir):
        log.print_debug("Directories already exist. Skipping download.")
        return
    log.print_debug(f".... Downloading the HAR DATASET {baseline_view_dir}")
    # Download the file
    url = 'https://zenodo.org/api/records/11992126/files-archive'
    file_name = 'archive.zip'
    response = requests.get(url)
    log.print_debug(f".... response {response}")

    # Save the downloaded content as a zip file
    with open(file_name, 'wb') as file:
        file.write(response.content)

    log.print_debug(f".... dataset write ok {response}")
    # Create 'data' folder if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Unzip the main archive to 'data' folder
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Files extracted to {extract_dir} successfully.")

    os.remove(file_name)
    
    # Paths to the two zip files inside 'data'
    standardized_view_zip = os.path.join(extract_dir, 'standardized_view.zip')
    baseline_view_zip = os.path.join(extract_dir, 'baseline_view.zip')

    # Function to unzip a file if it exists
    def unzip_file_if_exists(file_path, extract_to):
        if os.path.exists(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extracted: {file_path} to {extract_to}")
            os.remove(file_path)
        else:
            print(f"File {file_path} does not exist, skipping extraction.")

    # Unzip both files inside 'data', if they exist
    unzip_file_if_exists(standardized_view_zip, standardized_view_dir)
    unzip_file_if_exists(baseline_view_zip, baseline_view_dir)
