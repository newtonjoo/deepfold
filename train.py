# Copyright 2022 Deepfold Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training DeepFold protein structure prediction model."""

# OS & MPI config. please config before any import of jax / tf.
import os
import sys
import pickle
import socket
import jax.numpy as jnp
import signal
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from deepfold.train.train_config import train_config, custom_train_config

hostname = socket.gethostname()
gpus_per_node=8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--custom-config', type=str, default=None)
args = parser.parse_args()
if args.custom_config is not None:
  os.environ['custom_config'] = args.custom_config

config = train_config
if 'custom_config' in os.environ:
  try:
    config = custom_train_config(os.environ['custom_config'])
  except Exception as e:
    print (e)
use_mpi = config.global_config.use_mpi
if use_mpi:
  from mpi4py import MPI
  mpi_comm = MPI.COMM_WORLD
  mpi_rank = mpi_comm.Get_rank()
  is_main_process = (mpi_rank == 0)
  #os.environ['CUDA_VISIBLE_DEVICES'] = str(mpi_rank % train_config.global_config.gpus_per_node)
  os.environ['CUDA_VISIBLE_DEVICES'] = str(mpi_rank % gpus_per_node)
  if 'SLURM_JOB_NUM_NODES' in os.environ:
    if os.environ['SLURM_JOB_NUM_NODES'] == '1': # single node
      os.environ['CUDA_VISIBLE_DEVICES']=os.environ['SLURM_JOB_GPUS'].split(',')[mpi_rank]
else:         # assume single gpu is used.
  mpi_comm = None
  mpi_rank = 0
  is_main_process = True
# external import
from absl import logging
from multiprocessing import Queue

# internal import
from deepfold.model.config import model_config as get_model_config
from deepfold.train.data_system import DataSystem, GetBatchProcess
from deepfold.train.utils import get_queue_item
from deepfold.train.trainer import Trainer

def train(train_config):
  """
  main function of training (single gpu).
  """
  # get configs
  gc = train_config.global_config
  model_config = get_model_config(gc.model_name, is_training=True)
  
  # construct datasets
  logging.info("constructing train data ...")
  train_data = DataSystem(model_config, train_config.data.train)
  logging.info(f"data num : {train_data.num_prot}")
  logging.info("constructing validation data ...")
  try:
    eval_data = DataSystem(model_config, train_config.data.eval)
    logging.info(f"data num : {eval_data.num_prot}")
  except:
    logging.warning("failed to load validation data. poor configurations may be provided.")
    eval_data = None

  # create batch processes
  train_queue = Queue(gc.max_queue_size)
  train_batch_proc = GetBatchProcess(
      queue=train_queue,
      data=train_data,
      num_batches=(gc.end_step - gc.start_step)*gc.gradient_accumulation + 1,  # add 1 for the initialization batch
      is_training=True,
      random_seed=gc.random_seed,
      mpi_rank=mpi_rank)                            # pass rank to generate different batches among mpi.
  train_batch_proc.start()
  
  if eval_data is not None:
    eval_queue = Queue(gc.max_queue_size)
    eval_batch_proc = GetBatchProcess(
        queue=eval_queue,
        data=eval_data,
        num_batches=(gc.end_step - gc.start_step) // gc.eval_freq + 1,
        is_training=True,
        random_seed=gc.random_seed,
        mpi_rank=mpi_rank)                          # pass rank to generate different batches among mpi.
    eval_batch_proc.start()
    
  def proc_terminate():
    if train_batch_proc.is_alive():
      train_batch_proc.terminate()
    if eval_data is not None and eval_batch_proc.is_alive():
      eval_batch_proc.terminate()
    sys.exit()

  signal.signal(signal.SIGINT,proc_terminate)
  signal.signal(signal.SIGTERM,proc_terminate)
  signal.signal(signal.SIGCONT,proc_terminate)

  # define and initialize trainer
  trainer = Trainer(
      global_config=gc,
      optim_config=train_config.optimizer,
      model_config=model_config,
      mpi_comm=mpi_comm)
  logging.info("initializing ...")
  logging.error(f"running on {hostname}:{os.environ['CUDA_VISIBLE_DEVICES']}")

  def make_minibatch(batch_list):
    """ add batchs over first dim"""
    return {k: jnp.array([batch_list[i][k] for i in range(len(batch_list))]) for k in batch_list[0].keys()}

  def get_minibatch(data_queue, size=gc.minibatch_size):
    batch_list = []
    for i in range(size):
      rng, batch = get_queue_item(data_queue)
      batch_list.append(batch)
    return rng, make_minibatch(batch_list)

  _, init_batch = get_minibatch(train_queue, size=1)    # do NOT use the returned rng to initialize trainer.
  trainer.initialize(init_batch, load_format=gc.ckpt_format)
  
  # conduct training
  logging.info("training ...")
  for step in range(gc.start_step, gc.end_step):
    for i in range(1,gc.gradient_accumulation+1):
      update_rng, batch = get_minibatch(train_queue)
      if i < gc.gradient_accumulation:
        trainer.train_step(step, batch, update_rng, accumulation=True)
      else:
        trainer.train_step(step, batch, update_rng, silent=(not is_main_process))
    if eval_data is not None and trainer.is_eval_step(step):
      eval_rng, batch = get_minibatch(eval_queue)
      trainer.train_step(step, batch, eval_rng, silent=(not is_main_process), update=False)
  logging.info("finished training.")

  proc_terminate()


if __name__ == "__main__":
  LOG_VERBOSITY = {
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG
  }
  if is_main_process:
    logging.set_verbosity(LOG_VERBOSITY[train_config.global_config.verbose.upper()])
  else:
    logging.set_verbosity(logging.ERROR)
  train(config)

