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

import os
import io
import uuid
import torch
import uvloop
import hashlib
import asyncio
import tempfile
import aiofiles
import tempfile
import numpy as np
import aiobotocore
import botocore.config
from typing import List, Dict
from dotenv import dotenv_values
from types import SimpleNamespace
from aiobotocore.session import get_session

# Load environment variables
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')

# Configure the S3 client
client_config = botocore.config.Config(
    max_pool_connections=256,
)

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Define a semaphore to limit concurrent downloads (adjust as needed)
semaphore = asyncio.Semaphore(1000)

async def upload_mask(bucket: str, model: torch.nn.Module, mask: int, wallet: 'bt.wallet', compression: int):
    filename = f'mask-{mask}-{wallet.hotkey.ss58_address}.pt'
    model_state_dict = model.state_dict()
    indicies = await get_indicies_for_mask(model, mask, compression)
    
    for name, param in model.named_parameters():
        model_state_dict[name] = param.data.view(-1)[indicies[name].to(model.device)].cpu()
    
    # Create a temporary file and write the model state dictionary to it
    with tempfile.NamedTemporaryFile(delete=False) as module_buffer:
        torch.save(model_state_dict, module_buffer)
        temp_file_name = module_buffer.name  # Store the file name

    # Now open the temporary file for reading in binary mode
    with open(temp_file_name, 'rb') as f:
        session = get_session()
        async with session.create_client(
            's3',
            region_name='us-east-1',
            config=client_config,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        ) as s3_client:
            # Pass the standard file object to put_object
            await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
            await s3_client.put_object_acl(
                Bucket=bucket,
                Key=filename,
                GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
                GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
            )
    # Clean up the temporary file
    os.remove(temp_file_name)

async def upload_master( bucket, model: torch.nn.Module, wallet: 'bt.wallet' ):
    upload_filename = f'master-{wallet.hotkey.ss58_address}.pt'
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        async with aiofiles.tempfile.NamedTemporaryFile() as module_buffer:
            torch.save(model.state_dict(), module_buffer)
            await module_buffer.seek(0)  # Reset the buffer's position to the beginning.
            async with aiofiles.open(module_buffer.name, 'rb') as f:
                await s3_client.put_object(Bucket=bucket, Key=upload_filename, Body=f)
        await s3_client.put_object_acl(
            Bucket=bucket,
            Key=upload_filename,
            GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
            GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
        )

async def get_indicies_for_mask( model: torch.nn.Module, mask: int, compression: int ) -> Dict[ str, torch.LongTensor ]:
    result = {}
    rng = np.random.default_rng(int(hashlib.md5(str(mask).encode('utf-8')).hexdigest(), 16) % (2**32))
    for name, param in sorted(model.named_parameters()):
        indices = rng.choice(param.numel(), size=max(1, int(param.numel() // compression)), replace=False)
        result[name] = torch.from_numpy(indices).long().cpu()
    return result

async def download_file(s3_client, bucket, filename):
    async with semaphore:
        try:
            temp_file = os.path.join(tempfile.gettempdir(), f"{filename}")

            # Check if the file already exists
            if os.path.exists(temp_file):
                print(f"File {temp_file} already exists, skipping download.")
                return temp_file

            # First, get the object size
            head_response = await s3_client.head_object(Bucket=bucket, Key=filename)
            object_size = head_response['ContentLength']

            # Decide on the number of parts
            CHUNK_SIZE = 1 * 1024 * 1024
            part_size = CHUNK_SIZE  # CHUNK MB
            parts = []
            for i in range(0, object_size, part_size):
                start = i
                end = min(i + part_size - 1, object_size - 1)
                parts.append((start, end))

            # Function to download a part
            async def download_part(part_number, start, end):
                range_header = f"bytes={start}-{end}"
                response = await s3_client.get_object(
                    Bucket=bucket,
                    Key=filename,
                    Range=range_header
                )
                part_file = f"{temp_file}.part{part_number}"
                async with aiofiles.open(part_file, 'wb') as f:
                    async for chunk in response['Body'].iter_chunks(chunk_size=CHUNK_SIZE):
                        await f.write(chunk)
                return part_file

            # Download parts concurrently
            download_tasks = [
                download_part(idx, start, end)
                for idx, (start, end) in enumerate(parts)
            ]

            part_files = await asyncio.gather(*download_tasks)

            # Combine parts
            async with aiofiles.open(temp_file, 'wb') as outfile:
                for part_file in sorted(part_files, key=lambda x: int(x.split('part')[-1])):
                    async with aiofiles.open(part_file, 'rb') as infile:
                        while True:
                            chunk = await infile.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            await outfile.write(chunk)
                    # Remove the part file
                    os.remove(part_file)

            return temp_file
        except Exception as e:
            print(f'\t\tFailed to download mask {filename}: {e}')
            return None

async def handle_file(s3_client, bucket, filename, hotkey, mask):
    temp_file = await download_file(s3_client, bucket, filename)
    if temp_file:
        return SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=filename, mask=mask, temp_file=temp_file)
    return None

async def process_bucket(s3_client, bucket, masks):
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for mask in masks:
        async for page in paginator.paginate(Bucket=bucket, Prefix=f'mask-{mask}'):
            if 'Contents' not in page:
                continue
            download_tasks = []
            for obj in page.get('Contents', []):
                filename = obj['Key']
                try:
                    parts = filename.split('-')
                    file_mask = int(parts[1])
                    hotkey = parts[2].split('.')[0]
                    if file_mask == mask:
                        download_tasks.append(handle_file(s3_client, bucket, filename, hotkey, file_mask))
                except Exception as e:
                    print(f'Error processing filename {filename}: {e}')
                    continue
            results = await asyncio.gather(*download_tasks)
            files.extend([res for res in results if res])
    return files

async def download_files_for_buckets_and_masks(buckets: List[str], masks: List[int]) -> Dict[int, List[SimpleNamespace]]:
    session = get_session()
    # Use 'async with' to create the S3 client context
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        tasks = []
        for bucket in set(buckets):
            if bucket is None:
                continue
            tasks.append(process_bucket(s3_client, bucket, masks))
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists
        files = [item for sublist in results for item in sublist]
        
        # Create a dictionary with masks as keys and list of files as values
        mask_dict = {}
        for file in files:
            mask = file.mask
            if mask not in mask_dict:
                mask_dict[mask] = []
            mask_dict[mask].append(file)
        
        return mask_dict

async def get_files_for_mask_from_temp(mask: int) -> List[str]:
    temp_dir = tempfile.gettempdir()
    mask_files = []
    for filename in os.listdir(temp_dir):
        if filename.startswith(f"mask-{mask}-") and filename.endswith(".pt"):
            mask_files.append(os.path.join(temp_dir, filename))
    return mask_files
