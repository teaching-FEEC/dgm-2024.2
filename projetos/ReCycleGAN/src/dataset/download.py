"""Module to download and extract a ZIP file from a URL."""
from pathlib import Path
import zipfile
import requests

# URL of the ZIP file
NEXET = 'https://github.com/TiagoCAAmorim/dgm-2024.2/releases/download/v0.1.0-nexet/Nexet.zip'

def download_and_extract(url, path):
    """
    Download and extract a ZIP file from a URL.

    Args:
    - url: URL of the ZIP file.
    - path: Path to extract the ZIP file.
    """
    path = Path(path)
    zip_path = path / 'file.zip'
    response = requests.get(url, timeout=10)
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)

    zip_path.unlink()

if __name__ == '__main__':
    download_and_extract(NEXET, Path(__file__).parent)
