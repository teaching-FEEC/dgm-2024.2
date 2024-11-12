"""Module to download and extract a ZIP file from a URL."""
from pathlib import Path
import zipfile
import requests

# URL of the ZIP file
RELEASE = 'https://github.com/TiagoCAAmorim/dgm-2024.2/releases/download/v0.1.0-nexet/'

def download_and_extract(url, path, verbose=False):
    """
    Download and extract a ZIP file from a URL.

    Args:
    - url: URL of the ZIP file.
    - path: Path to extract the ZIP file.
    - verbose: Bool to print messages. Default is False.
    """
    path = Path(path)
    zip_path = path / 'file.zip'
    if verbose:
        print(f'Creating folder: {zip_path.parent}')
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'Downloading {Path(url).name}.')
    response = requests.get(url, timeout=10)
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    if verbose:
        print('Unpaking zip file.')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)

    if verbose:
        print('Deleting zip file.')
    zip_path.unlink()
    if verbose:
        print('Done.')

if __name__ == '__main__':
    # Downloads and extracts nexet images to 'usual' folder.
    out_folder = Path(__file__).parent.parent.parent / 'data2/external/nexet'
    for file in ['nexet.zip', 'output_cyclegan.zip', 'output_turbo.zip']:
        download_and_extract(RELEASE + file, out_folder, verbose=True)

    out_folder = Path(__file__).parent.parent.parent / 'data2/'
    download_and_extract(RELEASE + 'checkpoints.zip', out_folder, verbose=True)
