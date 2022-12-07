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
import signal
import lzma
import socket
import json

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from deepfold.train.train_config import train_config, custom_train_config

gpus_per_node=8


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
  mpi_size = mpi_comm.Get_size()
  is_main_process = (mpi_rank == 0)
  #os.environ['CUDA_VISIBLE_DEVICES'] = str(mpi_rank % train_config.global_config.gpus_per_node)
  os.environ['CUDA_VISIBLE_DEVICES'] = str(mpi_rank % gpus_per_node)
  if 'SLURM_JOB_NUM_NODES' in os.environ:
    if os.environ['SLURM_JOB_NUM_NODES'] == '1': # single node
      os.environ['CUDA_VISIBLE_DEVICES']=os.environ['SLURM_JOB_GPUS'].split(',')[mpi_rank]


else:         # assume single gpu is used.
  mpi_comm = None
  mpi_rank = 0
  mpi_size = 1
  is_main_process = True
# external import
from absl import logging
from multiprocessing import Queue, Pool
import jax
import jax.numpy as jnp
import jax.random as jrand

# internal import
from deepfold.model.config import model_config as get_model_config
from deepfold.train.utils import get_queue_item
from deepfold.train.trainer import Trainer

def build_distillation(train_config):
  """
  main function of building distillation dataset (single gpu).
  """
  from deepfold.train.data_system import DataSystem, GetBatchProcess
  # get configs
  gc = train_config.global_config
  model_config = get_model_config(gc.model_name, is_training=True)
  logging.info("constructing dataset for distillation..")
  
  # construct datasets
  logging.info("constructing train data ...")
  train_data = DataSystem(model_config,
          train_config.distillation.build, build_distillation=True)
  num_data = train_data.num_build
  logging.info(f"num_data : {num_data}")

  # create batch processes
  train_queue = Queue(gc.max_queue_size)
  train_batch_proc = GetBatchProcess(
      queue=train_queue,
      data=train_data,
      num_batches=num_data + 1,  # add 1 for the initialization batch
      is_training=True,
      random_seed=gc.random_seed,
      mpi_rank=mpi_rank)                            # pass rank to generate different batches among mpi.
  train_batch_proc.start()

  # define and initialize trainer
  trainer = Trainer(
      global_config=gc,
      optim_config=train_config.optimizer,
      model_config=model_config,
      return_representations=True,
      mpi_comm=mpi_comm)
  logging.info("initializing ...")
  _, init_batch = get_queue_item(train_queue)    # do NOT use the returned rng to initialize trainer.
  trainer.initialize(init_batch, load_format=gc.ckpt_format)
  
  # save batch and value in evaluation mode
  for step in range(num_data):
    update_rng, batch = get_queue_item(train_queue)
    #logging.info(dict_str(batch, 0))
    value, loss = trainer.build_distillation_step(step,
                    batch, update_rng, silent=(not is_main_process))
    ret = dict(batch=batch, value=value, loss=loss)
    name = "".join(chr(i) for i in batch['name'][0])
    filename = f'{name}_{batch["crop_start"][0]}.xz'
    logging.info(f'name: {name}, crop_start: {batch["crop_start"][0]} saved to {filename}')
    filepath = os.path.join(train_config.data.distillation.save_dir, filename)
    with lzma.open(filepath, 'wb') as fp:
        pickle.dump(ret, fp, protocol=4)

  if train_batch_proc.is_alive():
    train_batch_proc.terminate()
    return


def train_distillation(train_config):
  """
  main function of training from distillation (single gpu).
  """
  from deepfold.distillation.data_system import DataSystem
  #logging.set_verbosity(logging.INFO)
  # get configs
  gc = train_config.global_config
  dc = train_config.distillation
  model_config = get_model_config(gc.model_name, is_training=True)
  logging.info(f"train via distillation start. \n" \
      f"model_name : {gc.model_name}, minibatch_size : {gc.minibatch_size}, gradient_accumulation: {gc.gradient_accumulation}\n" \
      f"start_step : {gc.start_step}, end_step: {gc.end_step}\n" \
      f"eval_freq : {gc.eval_freq}, save_freq: {gc.save_freq}\n" \
      f"save_dir : {gc.save_dir}, load_dir: {gc.load_dir}\n" \
      f"mpi rank : {mpi_rank}, mpi size: {mpi_size}, DEVICE:{os.environ['CUDA_VISIBLE_DEVICES']}\n" )
  if 'SLURM_JOB_GPUS' in os.environ:
    logging.info(f"slurm gpus: {os.environ['SLURM_JOB_GPUS']}")

  # save train config and model config
  os.makedirs(gc.save_dir, exist_ok=True)
  with open(os.path.join(gc.save_dir, f'{gc.model_name}_config.json'), 'w') as f:
    json.dump({'train_config':train_config.to_json(),'model_config':model_config.to_json()}, f)
  
  # construct train datasets
  train_data = DataSystem(dc.train.train_data_list, dc.train.dssp_dir)
  logging.info(f"constructing train data... num data : {train_data.num_data}") 
  train_queue = Queue(gc.max_queue_size)
  train_pool = Pool(gc.data_num_workers, train_data.queue_writer, (train_queue,))
  train_pool.close()

  # construct eval datasets
  eval_data = DataSystem(dc.train.eval_data_list, dc.train.dssp_dir)
  logging.info(f"constructing eval data... num data : {eval_data.num_data}") 
  eval_queue = Queue(gc.max_queue_size)
  eval_pool = Pool(1, eval_data.queue_writer, (eval_queue,))
  eval_pool.close()

  def make_minibatch(batch_list):
    """ add batchs over first dim"""
    return {k: jnp.array([batch_list[i][k] for i in range(len(batch_list))]) for k in batch_list[0].keys()}

  def pool_terminate():
    train_pool.terminate()
    eval_pool.terminate()
    sys.exit()

  signal.signal(signal.SIGINT,pool_terminate)
  signal.signal(signal.SIGTERM,pool_terminate)
  signal.signal(signal.SIGCONT,pool_terminate)

  with jax.disable_jit():
    rng = jrand.PRNGKey(gc.random_seed)
    rng = jrand.fold_in(rng, mpi_rank)
  
  # define and initialize trainer
  trainer = Trainer(
      global_config=gc,
      optim_config=train_config.optimizer,
      model_config=model_config,
      mpi_comm=mpi_comm)
  logging.info("initializing ...")
  init_batch = get_queue_item(train_queue)    # do NOT use the returned rng to initialize trainer.
  # trainer init
  init_batch = make_minibatch([init_batch])
  trainer.initialize(init_batch, load_format=gc.ckpt_format)

  # conduct training
  logging.info("training ...")
  for step in range(gc.start_step, gc.end_step):
    for i in range(1,gc.gradient_accumulation+1):
      batch_list = [get_queue_item(train_queue) for i in range(gc.minibatch_size)]
      batch = make_minibatch(batch_list)
      rng, _ = jrand.split(rng)
      if i < gc.gradient_accumulation:
        trainer.train_step(step, batch, rng, accumulation=True)
      else:
        trainer.train_step(step, batch, rng, silent=(not is_main_process))
    if trainer.is_eval_step(step):
      batch_list = [get_queue_item(eval_queue) for i in range(gc.minibatch_size)]
      batch = make_minibatch(batch_list)
      trainer.train_step(step, batch, rng, silent=(not is_main_process), update=False)

  train_pool.terminate()
  eval_pool.terminate()
  logging.info("finished training.")

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
    logging.set_verbosity(LOG_VERBOSITY[config.global_config.verbose.upper()])
  else:
    logging.set_verbosity(logging.ERROR)
  if config.global_config.build_distillation:
      build_distillation(config)
  if config.global_config.train_distillation:
      train_distillation(config)

