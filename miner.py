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
import wandb
import torch
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
from dataset import SubsetFineWebEdu2Loader

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Miner:
    
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')    
        parser.add_argument('--project', type=str, default='220A', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=222, help='Bittensor network UID.')
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
        print('\n', '-' * 40, 'Config', '-' * 40)
        self.config = Miner.config()
        print(self.config)
        
        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid) 
        bt.logging.off()   
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise ValueError(f'Wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}')    
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address) 
        print('\n', '-' * 40, 'Objects', '-' * 40)  
        print(f'Wallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}') 
        
        # Init bucket.
        try: 
            if self.config.bucket != self.subtensor.get_commitment(self.config.netuid, self.uid): raise ValueError('')
        except: self.subtensor.commit(self.wallet, self.config.netuid, self.config.bucket)
        print('Bucket:', self.config.bucket)  
        
        # Init Wandb.
        if self.config.use_wandb:
            # Delete all runs with my name and create a new one.
            try: [run.delete() for run in wandb.Api().runs(path=self.config.project) if run.name == f'M{self.uid}' and print(f'Deleting old run: {run}')]
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.uid}', config=self.config)
            
        # Init model.
        print('\n', '-' * 40, 'Hparams', '-' * 40)
        self.hparams = load_hparams()
        self.model = LlamaForCausalLM( config = self.hparams.model_config )
        self.model.to(self.config.device)
        self.model.train()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr = self.hparams.learning_rate,  # Peak learning rate
            betas = ( self.hparams.optimizer_beta1, self.hparams.optimizer_beta2 ), # B1 and B2
            weight_decay = self.hparams.optimizer_weight_decay,  # Weight decay
            foreach = True,  # more memory usage, but faster
        )
        self.scheduler = CosineAnnealingLR( self.optimizer, T_max = self.hparams.cosine_epoch_length, eta_min=self.hparams.eta_min, last_epoch=-1 )
        
        # Init buckets.
        self.buckets = []
        for uid in tqdm(self.metagraph.uids):
            try: self.buckets.append('decis')
            except: self.buckets.append(None)
            
        # Init run state.
        self.current_mask = None
        self.current_block = None
        self.block_event = asyncio.Event()
        self.mask_event = asyncio.Event()
        self.stop_event = asyncio.Event()
        
    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(target=self.block_listener, args=( self.loop, ), daemon=True).start()
        self.tasks = [
            self.update(),
            self.download(),
            self.apply(),
            self.train()
        ]
        try:
            await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            self.stop_event.set()
            await asyncio.gather(*self.tasks)
            
    # Returns the mask window based on a block.
    def block_to_mask( self, block: int ) -> int:
        return int(block / self.hparams.mask_window_length)
    # A listener thread which posts the block event 
    # when the chain annouces a new block.
    def block_listener( self, loop ):
        print(f"[events]: Start listening.")
        def handler( event, _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.block_event.set)
            if self.block_to_mask(self.current_block) != self.current_mask:
                self.current_mask = self.block_to_mask(self.current_block)
                loop.call_soon_threadsafe(self.mask_event.set)
                print(f"[events]: New mask: {self.current_mask}")
            print(f"[events]: New block: {self.current_block}")
        # Subscribe to block headers with the custom handler
        bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
        
    # Helper for waiting on block time
    async def wait_for_new_block( self ):
        while True: 
            await self.block_event.wait(); 
            self.block_event.clear()
            
    # Helper for waiting on mask time.
    async def wait_for_new_mask( self ):
        while True: 
            await self.mask_event.wait(); 
            self.mask_event.clear()
        
    # Updates chain state on a background loop.
    async def update( self ):
        print (f'[update]: Started update loop.')
        # Forever until the stop event is set.
        while not self.stop_event.is_set():
            try:
                self.hparams = load_hparams()
                self.subtensor = bt.subtensor(config=self.config)
                self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
                if len(self.buckets) != len(self.metagraph.uids):
                    for uid in tqdm(self.metagraph.uids):
                        # try: buckets.append(subtensor.get_commitment(config.netuid, uid))
                        try: self.buckets.append('decis')
                        except: self.buckets.append(None)
                print (f'[update]: Updated state. Waiting {60} seconds.')
                await asyncio.sleep(60)  # Add a sleep to prevent tight loop
            except Exception as e:
                print(f'Error in update: {e}')
                traceback.print_exc()


    # Pulls masks from other peers on a background loop.  
    async def download( self ):
        print (f'[download]: Started download loop.')
        while not self.stop_event.is_set(): 
            try:
                print (f'[download]: Downloading masks for: {self.current_mask} ')
                files = await download_files_for_buckets_and_masks( buckets = self.buckets, masks = [self.current_mask] )
                print ('files', files)
                print (f'[download]: Downloaded {sum([len(files[k]) for k in files])} masks for: {self.current_mask} ')
                await asyncio.sleep(1)
            except Exception as e:
                print(f'Error in download: {e}')
                traceback.print_exc()
       
    # Apply masks to model.
    async def apply( self ):
        print (f'[apply]: Started apply loop.')
        while not self.stop_event.is_set():
            try:
                await self.wait_for_new_mask()
                indices = await get_indicies_for_mask( self.model, self.current_mask - 1, self.hparams.compression )
                mask_files = await get_files_for_mask_from_temp( mask = self.current_mask - 1 )
                print (f'[apply]: Loaded {len(mask_files)} masks for: {self.current_mask - 1} ')

                # Apply masks to model and average them.
                masks_per_param = { name: 0 for name in self.model.named_parameters() }
                for mask_info in mask_files:
                    try:
                        mask = torch.load(mask_info.temp_file, map_location=torch.device(self.model.device), weights_only=True)
                        for name, param in self.model.named_parameters():
                            if name not in indices: continue
                            if name not in mask: continue
                            values = mask[name].to(model.device)
                            indices = indices[name].to(model.device)
                            param.data.view(-1)[indices] += values 
                            masks_per_param[name] += 1
                            del values
                    except Exception as e:
                        pass
                for name, param in self.model.named_parameters():
                    if name not in masks_per_param: continue
                    if name not in indices: continue
                    if masks_per_param[name] == 0: continue # Nothing to average.
                    indices = indices[name].to(self.config.device)
                    param.data.view(-1)[indices] /= (masks_per_param[name] + 1)
                print (f'[apply]: Applied {len(mask_files)} masks for: {current_mask - 1}')
            except Exception as e:
                print(f'Error in apply: {e}')
                traceback.print_exc()
                
    # Trains for model.  
    async def train(self):
        print (f'[apply]: Started train loop.')
        # Forever until the stop event is set.
        while not self.stop_event.is_set():
            try:
                # Train for the current mask.
                print (f'[train]: Loading dataset')
                start_mask = self.current_mask
                pages = SubsetFineWebEdu2Loader.next_pages(
                    offset = self.current_block * self.hparams.pages_window_speed,
                    n_pages = 10,
                    seed = self.uid 
                )
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = self.config.actual_batch_size,
                    sequence_length = self.hparams.sequence_length,
                    pages_info = pages,
                    tokenizer = self.hparams.tokenizer
                )
                # Asynchronously initialize the dataset
                await dataset.initialize()
                print (f'[train]: Loaded dataset pages: {[p[1] for p in pages]}')

                # Train my model on the current page.
                print (f'[train]: Training on pages: {[p[1] for p in pages]}')
                torch.cuda.empty_cache() # Empty cache going into the training step.
                self.optimizer.zero_grad() # Clear any lingering grads.
                total_loss = 0.0
                total_steps = self.hparams.desired_batch_size // self.config.actual_batch_size
                for idx, batch in enumerate( dataset ):
                    input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                    labels = input_ids.clone()
                    labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                    with torch.amp.autocast( device_type = self.model.device.type, dtype = torch.bfloat16 ):  # Enable autocasting for mixed precision
                        outputs = self.model(input_ids = input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    loss = outputs.loss / (total_steps + 1) # Divide by number of accumulations.
                    loss.backward()
                    # Break on total steps of if we have 1 block left.
                    if idx >= total_steps - 1:
                        print ('[train]: Break training, no more steps.')
                        break
                    elif start_mask != self.current_mask:
                        print (f'[train]: Break training, new mask {start_mask} != {self.current_mask}')
                        break
                print (f'[train]: Trained {total_steps} steps with an average loss of {total_loss/(idx+1)}.')
                    
                # Apply step and clean memory.
                if self.hparams.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.grad_clip)
                self.optimizer.step()
                self.scheduler.step()  # Update the learning rate.
                self.optimizer.zero_grad()                        
                del input_ids, labels, outputs
                torch.cuda.empty_cache()

                # Upload our model mask to S3.
                print (f'[train]: Uploading for mask: {self.current_mask}')
                await upload_mask( self.config.bucket, self.model, self.current_mask, self.wallet, self.hparams.compression )
                print (f'[train]: Finished upload for mask: {self.current_mask}')
                    
            except Exception as e:
                print(f'Error in train: {e}')
                traceback.print_exc()

if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())
