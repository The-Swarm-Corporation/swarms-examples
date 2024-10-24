import os
from sec_edgar_downloader import Downloader
from loguru import logger
from datetime import datetime
from typing import List, Optional

# Initialize the SEC EDGAR Downloader with email and company name
dl = Downloader(
    email_address="kye@swarms.world", company_name="swarms"
)


def list_all_files_in_directory(directory: str) -> None:
    """
    Lists all files and folders in the specified directory for debugging purposes.

    Args:
        directory (str): The directory path to inspect.
    """
    if os.path.exists(directory):
        logger.info(f"Listing contents of directory: {directory}")
        for root, dirs, files in os.walk(directory):
            logger.info(f"Root: {root}")
            if dirs:
                logger.info(f"Directories: {dirs}")
            if files:
                logger.info(f"Files: {files}")
    else:
        logger.error(f"Directory not found: {directory}")


def get_current_quarter() -> int:
    """
    Determine the current quarter based on the current month.

    Returns:
        int: The current quarter (1, 2, 3, or 4).
    """
    current_month = datetime.now().month
    return (current_month - 1) // 3 + 1


def fetch_this_quarters_filings(
    cik: str, form_type: str = "10-Q"
) -> Optional[List[str]]:
    """
    Fetch filings for the current quarter for a specified company (CIK).

    Args:
        cik (str): Central Index Key (CIK) of the company.
        form_type (str): The type of filing to fetch (default is '10-Q').

    Returns:
        Optional[List[str]]: A list of file paths where the filings were downloaded, or None if an error occurs.
    """
    logger.info(
        f"Fetching {form_type} filings for CIK: {cik} for the current quarter..."
    )

    try:
        # Download the specified filings
        dl.get(form_type, cik)

        # Define the directory where the filings will be downloaded
        download_dir = os.path.join("sec-edgar-filings")

        # List all files and directories inside the download directory for debugging
        list_all_files_in_directory(download_dir)

        # Now move forward with searching inside the relevant folder
        company_dir = os.path.join(download_dir, cik, form_type)
        if not os.path.exists(company_dir):
            logger.error(
                f"Download directory not found: {company_dir}"
            )
            return None

        # Filter out filings from this quarter
        current_year = datetime.now().year
        current_quarter = get_current_quarter()
        current_quarter_filings = []

        # Walk through downloaded folders
        for root, dirs, files in os.walk(company_dir):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                filing_date = None

                for file in os.listdir(folder_path):
                    if file.endswith(".txt") or file.endswith(
                        ".html"
                    ):
                        with open(
                            os.path.join(folder_path, file),
                            "r",
                            encoding="utf-8",
                        ) as f:
                            content = f.read()
                            if "FILED AS OF DATE" in content:
                                filing_date_str = content.split(
                                    "FILED AS OF DATE"
                                )[1].strip()[:10]
                                filing_date = datetime.strptime(
                                    filing_date_str, "%Y-%m-%d"
                                )
                                break

                if filing_date:
                    filing_quarter = (filing_date.month - 1) // 3 + 1
                    if (
                        filing_date.year == current_year
                        and filing_quarter == current_quarter
                    ):
                        for file in os.listdir(folder_path):
                            if file.endswith(
                                ".html"
                            ) or file.endswith(".xml"):
                                file_path = os.path.join(
                                    folder_path, file
                                )
                                current_quarter_filings.append(
                                    file_path
                                )

        logger.info(
            f"Downloaded and filtered {len(current_quarter_filings)} filings for the current quarter."
        )
        return current_quarter_filings

    except Exception as e:
        logger.error(
            f"Failed to fetch filings for CIK: {cik}. Error: {e}"
        )
        return None


def main():
    cik = "320193"  # Apple's CIK
    form_type = "10-Q"  # Quarterly report

    # Fetch this quarter's filings
    downloaded_files = fetch_this_quarters_filings(
        cik=cik, form_type=form_type
    )

    if downloaded_files:
        logger.info(f"Downloaded files: {downloaded_files}")
    else:
        logger.error(f"Failed to download filings for CIK: {cik}.")


if __name__ == "__main__":
    main()
