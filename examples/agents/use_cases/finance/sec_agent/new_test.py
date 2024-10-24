import os
from typing import List
from loguru import logger
from sec_edgar_downloader import Downloader

# Initialize the SEC EDGAR Downloader with email and company name
dl = Downloader(
    email_address="kye@swarms.world",
    company_name="swarms",
    download_folder="appl",
)
# Get the five most recent 8-K filings for Apple
dl.get("10-K", "AAPL", limit=5)


def _get_all_files_with_content(directory: str = "appl") -> dict:
    """
    Recursively gets all files in the specified directory and its subdirectories,
    excluding specific directories, and returns a dictionary with file names as keys
    and their contents as values.

    :param directory: The directory to search for files.
    :return: A dictionary of file names and their contents.
    """
    all_files_content = {}
    for root, dirs, files in os.walk(directory):
        # Exclude specific directories
        dirs[:] = [d for d in dirs if d not in ["prompts", "tools"]]
        for file in files:
            full_path = os.path.join(root, file)
            with open(full_path, "r") as f:
                content = f.read()
            all_files_content[file] = content
            logger.info(
                f"Found file: {file} with content length: {len(content)}"
            )
    return all_files_content


out = _get_all_files_with_content()
print(out)
