# The MIT License (MIT)
# © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
from typing import List, Dict, Tuple
import requests
import numpy as np
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import asyncio
import aiohttp
from queue import Queue, Empty
import threading


class SubsetLoader(IterableDataset):
    """
    Base class for data-specific subset loader classes.
    """

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples

        self.num_rows_per_page = 100

        # Buffer to hold pages loaded from the API
        self.buffer = []

        # Buffer to hold pages already loaded into a batch
        self.used_buffer = []

        # Buffer to hold padded sequences
        self.padded_buffer = []

    def _get_pad_size(self, input_ids):
        """
        Get the number of tokens to be padded to the sample to match
        the max allowed sequence length.
        If sample packing is activated, then return 1
        """

        if self.pack_samples:
            return 1

        sample_size = len(input_ids)

        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder

        # Apply modulo again to guarantee a pad size of 0 if remainder is 0
        pad_size = pad_size % self.sequence_length

        return pad_size

    def _refill_padded_buffer(self):
        """
        This method pulls sequences from `self.buffer`, pads them, and pushes
        them to the `self.padded_buffer`.
        """

        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            input_ids = []

            # Search for EOS token index and cut the buffer at it.
            try:
                EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            except ValueError:
                # If EOS token is not found, take the entire buffer
                EOS_index = len(self.buffer) - 1

            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]

            self.used_buffer += input_ids

            # Add to padded buffer without the EOS token.
            input_ids_without_eos = (
                input_ids[:-1]
                if input_ids[-1] == self.tokenizer.eos_token_id
                else input_ids
            )
            self.padded_buffer += input_ids_without_eos

            # Pad
            pad_size = self._get_pad_size(input_ids=input_ids_without_eos)
            self.padded_buffer += [self.tokenizer.eos_token_id] * pad_size

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []

        # Pad and prepare data for batching
        self._refill_padded_buffer()

        return self

    def __next__(self):
        batch = []

        while True:
            # If we have enough data in padded_buffer, create batches
            while (
                len(self.padded_buffer) >= self.sequence_length
                and len(batch) < self.batch_size
            ):
                batch.append(self.padded_buffer[: self.sequence_length])
                self.padded_buffer = self.padded_buffer[self.sequence_length :]

            if len(batch) == self.batch_size:
                return np.stack(batch)

            # If padded_buffer is insufficient, try to refill it
            if len(self.padded_buffer) < self.sequence_length:
                if self.buffer:
                    self._refill_padded_buffer()
                elif not self.data_queue.empty():
                    # Get data from the queue
                    self.buffer.extend(self.data_queue.get())
                    self._refill_padded_buffer()
                elif self.stop_event.is_set():
                    if batch:
                        return np.stack(batch)
                    else:
                        raise StopIteration
                else:
                    # Wait for data to become available
                    time.sleep(0.1)
            else:
                if batch:
                    return np.stack(batch)
                else:
                    raise StopIteration


class SubsetFineWebEdu2Loader(SubsetLoader):
    """
    A class for loading and managing subsets of the FineWebEdu2 dataset.
    """

    # Class-level constants
    name: str = "HuggingFaceFW/fineweb-edu-score-2"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    # Configuration parameters
    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries
    num_rows_per_page: int = 100

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        pages_info=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
    ):
        # Call the parent class constructor
        super().__init__(
            batch_size, sequence_length, num_pages, tokenizer, pack_samples
        )

        # Fetch dataset configurations
        self.configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()

        # Initialize buffers and tokenizer
        self.padded_buffer = []
        self.tokenizer = tokenizer

        # Initialize the stop_event here so it's always available
        self.stop_event = threading.Event()


        # Handle initialization based on pages_info
        if pages_info is not None:
            # Set pages and num_pages if pages_info is provided
            self.pages = pages_info
            self.num_pages = len(self.pages)
            # Initialize buffer
            self.buffer = []
            # Do not fetch data here
            # Data will be fetched in the initialize method

        else:
            # Initialize for asynchronous fetching
            self.pages = None
            self.num_pages = num_pages
            self.data_queue = Queue(maxsize=1)
            self.loop = asyncio.new_event_loop()
            # Start background fetching thread
            self.fetch_thread = threading.Thread(
                target=self._start_event_loop, daemon=True
            )
            self.fetch_thread.start()

    async def initialize(self):
        """Asynchronously fetch data after object creation."""
        await self.fetch_data_for_pages(self.pages)
        # Set stop event to indicate completion
        self.stop_event.set()

    def _start_event_loop(self):
        """
        Start the asynchronous event loop for background data fetching.
        """
        # Set the event loop for the current thread
        asyncio.set_event_loop(self.loop)
        # Run the asynchronous background fetch method
        self.loop.run_until_complete(self._async_background_fetch())

    async def _async_background_fetch(self):
        """
        Asynchronously fetches data in the background and adds it to the data queue.
        """
        # Continue fetching until stop event is set
        while not self.stop_event.is_set():
            # Check if specific pages are provided
            if self.pages is not None:
                # Fetch data for specified pages
                pages_to_fetch: List[Tuple[str, int, str]] = self.pages
                await self.fetch_data_for_pages(pages_to_fetch)
                # Put fetched data in the queue
                self.data_queue.put(self.buffer)
                # Log fetching information
                num_pages_fetched: int = len(pages_to_fetch)
                print(f"[INFO] Fetched {num_pages_fetched} pages.")
                # Stop fetching
                self.stop_event.set()
            else:
                # Fetch random pages if no specific pages are provided
                pages_to_fetch: List[Tuple[str, int, str]] = self.get_random_pages(
                    num_pages=self.num_pages
                )
                await self.fetch_data_for_pages(pages_to_fetch)
                self.data_queue.put(self.buffer)
                self.buffer = []
                await asyncio.sleep(0)

    @staticmethod
    def fetch_dataset_configs() -> Dict[str, Dict]:
        """
        Fetches the dataset configurations, including the number of rows and splits.
        """
        # Set up request parameters
        params = {"dataset": SubsetFineWebEdu2Loader.name}

        # Initialize retry attempt counter
        attempt = 0
        # Retry loop
        while attempt < SubsetFineWebEdu2Loader.retry_limit:
            try:
                # Send GET request to fetch dataset size information
                response = requests.get(
                    SubsetFineWebEdu2Loader.size_base_url, params=params
                )
                # Raise an exception for bad status codes
                response.raise_for_status()

                # Extract the configs dictionary from the response
                configs_dict = response.json()["size"]["splits"]

                # Create a dictionary with config names as keys and number of rows as values
                configs_data = {
                    entry["config"]: {
                        "num_rows": entry["num_rows"],
                        "split": entry["split"],
                    }
                    for entry in configs_dict
                    if entry["config"] != "default"
                }

                # Return the processed config data
                return configs_data

            except requests.exceptions.RequestException as e:
                # Handle request exceptions
                attempt += 1
                if attempt < SubsetFineWebEdu2Loader.retry_limit:
                    # Wait before retrying
                    time.sleep(SubsetFineWebEdu2Loader.retry_delay)
                else:
                    # Raise the exception if all retries are exhausted
                    raise

    @staticmethod
    def next_pages(offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100):
        """
        Generate a sequence of random pages based on a given seed and offset.
        """
        # Fetch the dataset configurations
        configs_data = SubsetFineWebEdu2Loader.fetch_dataset_configs()

        # Initialize the random number generator with the given seed
        rng = np.random.default_rng(hash(seed) & 0xFFFFFFFF)

        # Advance the random number generator by the offset
        rng.bit_generator.advance(offset)

        # Initialize the result list
        result = []

        # Generate n_pages random pages
        for _ in range(n_pages):
            # Choose a random config
            config = rng.choice(list(configs_data.keys()))

            # Choose a random page number within the config's range
            choice = rng.integers(
                0, configs_data[config]["num_rows"] - 1 - num_rows_per_page
            )

            # Append the chosen page to the result
            result.append((str(config), int(choice), configs_data[config]["split"]))

        # Return the list of generated pages
        return result

    async def fetch_data_for_pages(self, pages):
        """
        Asynchronously fetch data for multiple pages.
        """
        # Clear the buffer before fetching new data
        self.buffer = []

        # Use an asynchronous context manager for the client session
        async with aiohttp.ClientSession() as session:
            # Create a list of tasks for fetching data from each page
            tasks = [self._fetch_data_for_page(session, page) for page in pages]

            # Gather results from all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process the results
            for result in results:
                if isinstance(result, Exception):
                    # Handle exceptions if necessary
                    print(f"[ERROR] Exception occurred: {result}")
                else:
                    # Extend the buffer with fetched data
                    self.buffer.extend(result)

        # Log information about the fetch operation
        print(f"[INFO] Fetched {len(pages)} pages asynchronously.")
        print(f"[DEBUG] Total tokens fetched: {len(self.buffer)}")

    async def _fetch_data_for_page(
        self, session: aiohttp.ClientSession, page: Tuple[str, int, str]
    ) -> List[int]:
        """
        Asynchronously fetches data for a single page.
        """
        # Unpack the page tuple
        config_name, offset, split = page

        # Prepare the request parameters
        params = {
            "dataset": self.name,
            "config": config_name,
            "split": split,
            "offset": offset,
            "limit": self.num_rows_per_page,
        }

        # Initialize the attempt counter
        attempt = 0

        # Retry loop
        while attempt < self.retry_limit:
            try:
                # Make an asynchronous GET request
                async with session.get(self.rows_base_url, params=params) as response:
                    # Raise an exception for bad status codes
                    response.raise_for_status()

                    # Parse the JSON response
                    json_response = await response.json()

                    # Initialize a list to store token IDs
                    input_ids_list = []

                    # Process each row in the response
                    for row in json_response.get("rows", []):
                        content = row["row"].get("text", "")
                        if content:
                            # Encode the content and add EOS token
                            input_ids = self.tokenizer(content, truncation=True)[
                                "input_ids"
                            ]
                            input_ids.append(self.tokenizer.eos_token_id)
                            input_ids_list.extend(input_ids)
                        else:
                            # Log a warning for empty content
                            print(f"[WARNING] Empty content in row: {row}")

                    # Return the list of token IDs
                    return input_ids_list

            except aiohttp.ClientError as e:
                # Handle client errors (e.g., network issues)
                attempt += 1
                print(
                    f"[WARNING] Failed to fetch page {page} (attempt {attempt}/{self.retry_limit}): {e}"
                )
                if attempt < self.retry_limit:
                    # Wait before retrying
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Log the error after all attempts fail
                    print(
                        f"[ERROR] Failed to fetch page {page} after {self.retry_limit} attempts."
                    )
                    return []

            except Exception as e:
                # Handle unexpected exceptions
                print(f"[ERROR] Unexpected exception: {e}")
                return []

    def get_random_pages(self, num_pages):
        """
        Randomly sample pages from the dataset.
        """
        # Initialize an empty list to store the sampled pages
        pages = []

        # Iterate 'num_pages' times to sample pages
        for _ in range(num_pages):
            # Choose a random config name from the available configs
            config_name = np.random.choice(list(self.configs_data.keys()))

            # Choose a random page number within the range of available rows
            page = np.random.randint(
                0,
                self.configs_data[config_name]["num_rows"] - 1 - self.num_rows_per_page,
            )

            # Get the split for the chosen config
            split = self.configs_data[config_name]["split"]

            # Append the sampled page info as a tuple to the pages list
            pages.append((config_name, page, split))

        # Return the list of sampled pages
        return pages

    def __del__(self):
        """
        Cleanup method called when the object is about to be destroyed.
        """
        # Check if stop_event attribute exists and set it
        if hasattr(self, "stop_event"):
            self.stop_event.set()

        # Check if the event loop exists and is running
        if hasattr(self, "loop") and self.loop.is_running():
            # Stop the event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            # Wait for the fetch thread to complete
            self.fetch_thread.join()
            # Close the event loop
            self.loop.close()
