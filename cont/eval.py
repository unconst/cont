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
import sys
import json
import uuid
import time
import torch
import wandb
import boto3
import shutil
import argparse
import tempfile
import traceback
import bittensor as bt
from hparams import load_hparams
from types import SimpleNamespace
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

def main(config):
    bt.logging.off() # Turn of bt logging.
    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        # Check for existing runs with the same name and delete them
        try:
            api = wandb.Api()
            runs = api.runs(path=config.project)
            for run in runs:
                if run.name == f'Eval':
                    print(f'Deleting old run: {run}')
                    run.delete()
        except: pass
        run = wandb.init(project=config.project, resume='allow', name='Eval', config=config)

    while True:
        
        print('Loading chain state.')
        hparams = load_hparams()
        subtensor = bt.subtensor(config=config)
        metagraph = subtensor.metagraph(config.netuid)

        print('Iterating miners')
        for uid in metagraph.uids:
            # Check if we are evaling a specific miner.
            if config.uid is not None:
                uid = config.uid
            # Try to eval.
            try:
                print("Getting commitment from subtensor...")
                try:
                    bucket = subtensor.get_commitment(config.netuid, uid)
                except:
                    print ('Miner has no registered bucket. Continuing.')
                    time.sleep(1)
                    continue

                print(f"Preparing to download model state dict for UID {uid}...")
                filename = f"master-{metagraph.hotkeys[uid]}.pt"
                temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")

                print(f"Downloading file {filename} from bucket {bucket}...")
                # Initialize the S3 client (assuming AWS S3)
                CLIENT = boto3.client('s3')
                try:
                    CLIENT.download_file(bucket, filename, temp_file)
                except Exception as e:
                    print(f"No master for UID {uid}. Error: {e}")
                    time.sleep(1)
                    continue

                print("Loading model state dict...")
                model = LlamaForCausalLM(config=hparams.model_config)
                model_state_dict = torch.load(temp_file, map_location='cpu', weights_only = True)
                model.load_state_dict(model_state_dict)

                # Delete the temp file after loading the model state dict
                os.remove(temp_file)

                model_save_path = f'models/{uid}'
                if os.path.exists(model_save_path):
                    print(f"Deleting existing model at {model_save_path}...")
                    shutil.rmtree(model_save_path)

                print(f"Saving model to models/{uid}...")
                os.makedirs(model_save_path, exist_ok=True)
                model.save_pretrained(model_save_path)
                
                print(f"Saving tokenizer to models/{uid}...")
                hparams.tokenizer.save_pretrained(model_save_path)

                print("Running lm-eval harness...")
                lm_eval_command = (
                    f"lm-eval "
                    f"--model hf "
                    f"--model_args pretrained=./models/{uid},tokenizer=./models/{uid} "
                    f"--tasks {config.tasks} "
                    f"--device {config.device} "
                    f"--batch_size {config.actual_batch_size} "
                    f"--output_path models/{uid}/results "
                )
                print(f"Executing command: {lm_eval_command}")
                exit_code = os.system(lm_eval_command)
                if exit_code != 0:
                    print(f"Command eval script failed with exit code {exit_code}. Error: {os.strerror(exit_code)}. Continuing...")
                    continue

                print("Loading evaluation results...")
                results_dir = f"models/{uid}/results/.__models__{uid}/"
                latest_file = max([os.path.join(results_dir, f) for f in os.listdir(results_dir)], key=os.path.getctime)
                with open(latest_file, "r") as f:
                    results = json.load(f)

                print("Processing results...")
                for task_name, task_results in results['results'].items():
                    if task_name == 'winogrande':
                        metric_name = 'acc,none'
                    else:
                        metric_name = 'acc_norm,none'
                    metric_value = float(task_results.get(metric_name))
                    if metric_value is not None:
                        print(f"{uid}/{task_name}:  {metric_value}")
                        if config.use_wandb:
                            wandb.log({f"{task_name}": metric_value})
                    else:
                        print(f"{uid} - {task_name} not found in results")

                # Delete the model after running the eval off the device
                del model
                torch.cuda.empty_cache()

                # Delete the storage of the model
                shutil.rmtree(model_save_path)

            # Error in eval loop.
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Exiting gracefully.")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--project', type=str, default='220A', help='Optional wandb project name')
    parser.add_argument('--netuid', type=int, default=220, help='Bittensor network UID.')
    parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--uid', type=int, default=None, help='The miner to eval. If None, eval all miners.')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--tasks', type=str, default='arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag', help='Comma-separated list of tasks to evaluate')
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.network = 'test'
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
    main(config)