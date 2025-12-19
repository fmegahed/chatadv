# This script scrapes the contents of the URLs provided in the Google Doc and 
# the URLs from the Farmer School of Business Bulletin. The contents are saved
# in multiple pickle files for future use.

import os
import time
import random
import requests
import tempfile
from pathlib import Path

import pandas as pd

from langchain_community.document_loaders import SeleniumURLLoader
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from dotenv import load_dotenv
from openai import OpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


# Bulletin URLs for the Farmer School of Business
# ------------------------------------------------------------------------------

# URL of the webpage
url = "https://bulletin.miamioh.edu/farmer-business/"

# Send a request to the webpage
response = requests.get(url)

# Parse the HTML content of the webpage
soup = BeautifulSoup(response.content, 'html.parser')

# Find all 'a' tags within the specified CSS selector
links = soup.select('#degreesandprogramstextcontainer > ul > li > a')

# Extract the href attribute and convert to absolute URL
absolute_urls = [urljoin(url, link['href']) for link in links]


# URLs from the CSV
# ------------------
urls_from_csv = (
    pd.read_csv("https://raw.githubusercontent.com/fmegahed/chatadv/refs/heads/main/data/scraped_urls_revised.csv")
    ['url']
    .tolist()
)


# Combining the URLs (focusing on the CSV)
# ----------------------------------------
urls = urls_from_csv + absolute_urls


# Scraping the Contents of the URLs
# ------------------------------------------------------------------------------
loader = SeleniumURLLoader(urls = urls)
data = loader.load()



# Saving the data to our OpenAI vector store
# -------------------------------------------
load_dotenv(override = True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
VECTOR_STORE_ID = os.getenv('VECTOR_STORE_ID')

client = OpenAI(api_key = OPENAI_API_KEY)

# Throttling
def upload_with_backoff(path, *, max_attempts=8, base_delay=1.0, max_delay=60.0):
    """
    Upload a file with exponential backoff + jitter on transient failures.
    Retries on: 500s, timeouts, connection errors, and rate limits.
    """
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            with open(path, "rb") as f:
                return client.files.create(file=f, purpose="assistants")
        except (InternalServerError, APITimeoutError, APIConnectionError, RateLimitError) as e:
            last_err = e

            if attempt == max_attempts:
                raise

            # Exponential backoff with jitter
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay + random.uniform(0, 0.5 * delay)

            print(
                f"Upload failed ({type(e).__name__}) attempt {attempt}/{max_attempts}. "
                f"Retrying in {delay:.1f}s"
            )
            time.sleep(delay)

    raise last_err


# Delete all previous files in the data store:
def clear_vector_store(vector_store_id: str, *, also_delete_underlying_files: bool = False) -> None:
    next_page = None

    while True:
        page = client.vector_stores.files.list(
            vector_store_id=vector_store_id,
            limit=100,
            after=next_page,
        )

        for vs_file in page.data:
            # Remove from the vector store
            client.vector_stores.files.delete(
                vector_store_id=vector_store_id,
                file_id=vs_file.id,
            )

            # Optional: also delete the underlying File object (saves storage)
            if also_delete_underlying_files:
                # Depending on SDK object shape, the underlying file id may be vs_file.file_id
                underlying_file_id = getattr(vs_file, "file_id", None)
                if underlying_file_id:
                    client.files.delete(underlying_file_id)

        next_page = getattr(page, "last_id", None)
        if not next_page:
            break

clear_vector_store(VECTOR_STORE_ID, also_delete_underlying_files=True)

# Upload the files sequentially to the vectorstore:
def upload_files_sequentially(data):
    file_ids = []

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        for i, document in enumerate(data):
            title = document.metadata.get("title", "")
            source = document.metadata.get("source", "")
            body = document.page_content or ""

            md_text = (
                f"# Document {i}\n"
                f"## Title: {title}\n"
                f"## Source: {source}\n"
                f"## Contents:\n"
                f"{body}\n"
            )

            path = td_path / f"doc_{i:06d}.md"
            path.write_text(md_text, encoding="utf-8")

            with open(path, "rb") as f:
                created = upload_with_backoff(path)

            file_ids.append(created.id)
            time.sleep(0.1)

            if (i + 1) % 25 == 0:
                print(f"Uploaded {i + 1} files")
                time.sleep(5)

    return file_ids

file_ids = upload_files_sequentially(data)

# Attached uploaded files to the vector store in batches
def attach_files_to_vector_store(
    vector_store_id: str,
    file_ids: list[str],
    batch_size: int = 100,
):
    for start in range(0, len(file_ids), batch_size):
        chunk = file_ids[start : start + batch_size]

        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store_id,
            file_ids=chunk,
        )

        print(
            f"Attached files {start}–{start + len(chunk) - 1} | "
            f"status={batch.status} | counts={batch.file_counts}"
        )

attach_files_to_vector_store(VECTOR_STORE_ID, file_ids)
