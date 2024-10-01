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
# fmt: off

# Global imports.
import os
import sys 
import time
import math
import wandb
import torch
import asyncio
import argparse
import threading
import traceback
from tqdm import tqdm
import bittensor as bt
from typing import List
import torch.optim as optim
from dotenv import dotenv_values
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import local files.
from common import *
from hparams import load_hparams
from dataset import AsyncSubsetFineWebEdu2Loader

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Miner:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--project', type=str, default='220A', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=220, help='Bittensor network UID.')
        parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        config.subtensor.network = 'test'
        config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
        return config

    def __init__(self):
        # Init config.
        self.config = Miner.config()
        logger.info('\n' + '-' * 40 + ' Config ' + '-' * 40)
        logger.info(self.config)

        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise ValueError(f'Wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}')
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info('\n' + '-' * 40 + 'Objects' + '-' * 40)
        logger.info(f'\nWallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}')

        # Init bucket.
        try:
            if self.config.bucket != self.subtensor.get_commitment(self.config.netuid, self.uid):
                raise ValueError('')
        except:
            self.subtensor.commit(self.wallet, self.config.netuid, self.config.bucket)
        logger.info('Bucket:' + self.config.bucket)

        # Init Wandb.
        if self.config.use_wandb:
            # Delete all runs with my name and create a new one.
            try:
                [run.delete() for run in wandb.Api().runs(path=self.config.project)
                 if run.name == f'M{self.uid}' and logger.info(f'Deleting old run: {run}')]
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.uid}', config=self.config)

        # Init model.
        logger.info('\n' + '-' * 40 + 'Hparams' + '-' * 40)
        self.hparams = load_hparams()
        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)
        self.model.train()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,  # Peak learning rate
            betas=(self.hparams.optimizer_beta1, self.hparams.optimizer_beta2),  # B1 and B2
            weight_decay=self.hparams.optimizer_weight_decay,  # Weight decay
            foreach=True,  # more memory usage, but faster
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.cosine_epoch_length,
            eta_min=self.hparams.eta_min, last_epoch=-1
        )

        # Init buckets.
        self.buckets = []
        for uid in tqdm(self.metagraph.uids):
            try:
                self.buckets.append('decis')
            except:
                self.buckets.append(None)

        # Init run state.
        self.global_step = 0
        self.last_window = 0
        self.optimal_pages_per_step = 2.0
        self.current_block = self.subtensor.block
        self.current_window = self.block_to_window( self.current_block )
        self.new_block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()        
        print ( self.hparams )

    async def run(self):
        # Main loop.
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()
        while True:

            try:
                # Wait until we are on a new window.
                global_step_start_time = time.time()
                while self.current_window == self.last_window:
                    await asyncio.sleep(0.1)
                self.last_window = self.current_window
                
                # Start step.
                logger.info('\n' + '-' * 40 + f' Step{self.global_step} ' + '-' * 40)
                logger.info(f"Step: {self.global_step}, Window: {self.current_window}, "
                            f"Block: {self.current_block}, Time: {int(time.time())}")
                self.global_step += 1

                # Download files.    
                logger.info(f"\tDownloading slices for windows: {[self.current_window-1, self.current_window]}")
                start_time = time.time()
                files = await download_files_for_buckets_and_windows(buckets=self.buckets, windows=[self.current_window-1, self.current_window])
                downloaded_per_step = sum([len(files[k]) for k in files])
                logger.info(f"\t\tDownloaded {downloaded_per_step} slices for windows: {[self.current_window-1, self.current_window]} in {time.time() - start_time} seconds")
                
                # Apply slices to the model from the previous window.
                logger.info(f"\tApplying slices from window: {self.current_window - 1} to model.")
                start_time = time.time()
                slice_files = await apply_window_slices_to_model( model = self.model, window = self.current_window - 1, compression = self.hparams.compression)
                applied_per_step = len(slice_files)
                logger.info(f"\t\tApplied {applied_per_step} slices to model in {time.time() - start_time} seconds")
                
                # Train for the current window.
                logger.info(f"\tLoading {int(math.ceil(self.optimal_pages_per_step))} page dataset")
                start_time = time.time()
                start_window = self.current_window
                pages = await AsyncSubsetFineWebEdu2Loader.next_pages(
                    offset = self.current_block * self.hparams.pages_window_speed,
                    n_pages = int(math.ceil(self.optimal_pages_per_step)),
                    seed = self.uid
                )
                dataset = await AsyncSubsetFineWebEdu2Loader.create(
                    batch_size = self.config.actual_batch_size,
                    sequence_length = self.hparams.sequence_length,
                    pages_info = pages,
                    tokenizer = self.hparams.tokenizer
                )
                pages_per_step = len(pages)
                logger.info(f"\t\tLoaded dataset pages: {[p[1] for p in pages]} in {time.time() - start_time} seconds")

                # Train the model on the current page.
                logger.info(f"\tTraining on pages: {[p[1] for p in pages]}")
                start_time = time.time()
                torch.cuda.empty_cache()  # Empty cache going into the training step.
                self.optimizer.zero_grad()  # Clear any lingering grads.
                total_loss = 0.0
                total_steps = self.hparams.desired_batch_size // self.config.actual_batch_size
                window_exhuasted = False
                for idx, batch in enumerate(dataset):
                    input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                    labels = input_ids.clone()
                    labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                    with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                        outputs = self.model(input_ids=input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    loss = outputs.loss / (total_steps + 1)  # Divide by number of accumulations.
                    loss.backward()
                    if start_window != self.current_window:
                        window_exhuasted = True
                        break
                    
                # Update training pages based on window exhuastion.
                if window_exhuasted:
                    # Did exhuast window during training, decrease number of pages.
                    logger.info(f"\t\tExhuasted window during training.")
                    self.optimal_pages_per_step = max(1, self.optimal_pages_per_step * 0.9 )
                else: 
                    # Did not exhuast window during training.
                    # Waits until window is over.
                    logger.info(f"\t\tExhuasted dataset during training.")
                    self.optimal_pages_per_step *= 1.1

                # Apply step and clean memory.
                if self.hparams.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.grad_clip)
                self.optimizer.step()
                self.scheduler.step()  # Update the learning rate.
                self.optimizer.zero_grad()
                del input_ids, labels, outputs
                torch.cuda.empty_cache()

                # Calculate, print and log average loss
                average_loss = total_loss / (idx + 1)
                total_time = time.time() - start_time
                tokens_per_step = self.hparams.sequence_length * self.config.actual_batch_size * (idx + 1)
                tokens_per_second =  tokens_per_step / total_time
                logger.info(f"\t\tLoss: {average_loss}, learning_rate: {self.scheduler.get_last_lr()[0]}")
                logger.info(f"\t\tTraining completed in {total_time} seconds, Tokens per step: {tokens_per_step}, Tokens per second: {tokens_per_second}")

                # Upload our model slice to S3.
                logger.info(f"\tUploading for window: {self.current_window}")
                start_time = time.time()
                await upload_slice_for_window(self.config.bucket, self.model, self.current_window, self.wallet, self.hparams.compression)
                logger.info(f"\t\tFinished upload for window: {self.current_window} in {time.time() - start_time} seconds.")
                
                # Delete lingering files 
                logger.info(f"\tCleaning space.")
                start_time = time.time()
                await delete_files_before_window( window_max = self.current_window - self.hparams.max_history )
                await delete_files_from_bucket_before_window( bucket = self.config.bucket, window_max = self.current_window - self.hparams.max_history )
                logger.info(f"\t\tFinished cleaning space in {time.time() - start_time} seconds.")

                # Calculate and log global steps per second
                seconds_per_step = time.time() - global_step_start_time
                steps_per_second = 1 / seconds_per_step
                if self.config.use_wandb:
                    wandb.log({
                        "step_loss": average_loss,
                        "tokens_per_step": tokens_per_step,
                        "tokens_per_second": tokens_per_second,
                        "applied_per_step": applied_per_step,
                        "pages_per_step": pages_per_step,
                        "downloaded_per_step": downloaded_per_step,
                        "incentive": float(self.metagraph.I[self.uid]),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "seconds_per_step": seconds_per_step,
                        "steps_per_second": steps_per_second,
                        "tokens_per_global_step": tokens_per_second / seconds_per_step 
                    })
                print (f'\nGlobal step completed in {seconds_per_step} seconds\n')
                
            # Catch keyboard interrrupt.
            except KeyboardInterrupt:
                logger.info("Training interrupted by user. Stopping the run.")
                sys.exit(0)
            
            # Catch unknown.
            except Exception as e:
                logger.exception(f"Exception during training loop: {e}")
                continue

    # Returns the slice window based on a block.
    def block_to_window(self, block: int) -> int:
        return int(block / self.hparams.mask_window_length)

    # A listener thread which posts the block event
    # when the chain announces a new block.
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.new_block_event.set)
            if self.block_to_window(self.current_block) != self.current_window:
                self.current_window = self.block_to_window(self.current_block)
                loop.call_soon_threadsafe(self.new_window_event.set)
                logger.info(f"\t\tNew window: {self.current_window}")
        # Subscribe to block headers with the custom handler
        bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
            
if __name__ == "__main__":
    asyncio.run(Miner().run())
