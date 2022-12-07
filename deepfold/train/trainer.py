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

"""Container (Trainer) for training deepfold."""

# major imports
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import os
import time


# following packages for mpi communication will be just-in-time imported if use_mpi is True
# from mpi4py import MPI
# import mpi4jax
# from jax.tree_util import tree_flatten, tree_unflatten 

# import type name specifications
from deepfold.model.features import FeatureDict
from ml_collections import ConfigDict
from typing import Optional

# import major classes & functions
from deepfold.model.modules import AlphaFold
from deepfold.model.modules import AlphaFoldBatch
from deepfold.train.data_system import cast_to_precision
from deepfold.train.optimizer import Optimizer
from deepfold.train.mixed_precision import normalize_precision, set_deepfold_policy
from deepfold.train.tensorboard import init_writer, add_scalar_log


class Trainer:
  """
  main class to train the deepfold model.
  """
  def __init__(
      self,
      global_config: ConfigDict,
      optim_config: ConfigDict,
      model_config: ConfigDict,
      return_representations=False,
      **kwargs):

    self.gc = global_config

    self.precision = normalize_precision(self.gc.precision)
    set_deepfold_policy(self.precision)
    
    def _forward_fn(batch):
      model = AlphaFoldBatch(model_config.model, self.gc)
      return model(
          batch,
          is_training=True,
          compute_loss=True,
          ensemble_representations=False,
          return_representations=return_representations)

    self._init_fn = hk.transform(_forward_fn).init
    self._apply_fn = hk.transform(_forward_fn).apply
    #self._init_fn = hk.transform(_forward_fn).init
    #self._apply_fn = hk.transform(_forward_fn).apply
    self._loss_fn = None        # has to be initialized by external call on `Trainer.initialize()`
    self._grad_fn = None      # has to be initialized by external call on `Trainer.initialize()`
    self._grads = None        # for gradient accumulation
    self._loss = None

    # optimizer variables, have to be initialized by external call on `Trainer.initialize()`
    self.optim_config = optim_config
    self.optimizer = None       # instance of deepfold.trainer.optimizer.Optimizer
    self.optim_state = None     # optimizer state, key variable of maintaining model parameters

    # logging variables organized in format [(step, loss, time), (step, loss, time), ...]
    self.train_losses = []
    self.eval_losses = []

    # timing variables
    self._tic = time.time()      # tic & toc style for timing. handled in `Trainer.logging_on_the_fly`

    # mpi variables
    self.mpi_rank = 0
    if self.gc.use_mpi:
      self.mpi_comm = kwargs['mpi_comm']
      self.mpi_rank = self.mpi_comm.Get_rank()
      self.world_size = self.mpi_comm.Get_size()
    if self.mpi_rank == 0:
      init_writer()

    # path formatters of ckpts and loss curves
    self.auto_ckpt_name = \
        lambda step, format: f"{self.gc.model_name}_{step:05d}.{format}"
    self.auto_curve_name = \
        lambda is_train: f"{self.gc.model_name}_{'train' if is_train else 'eval'}_curve.npy"
    
    # step specifier
    self.is_logging_step = lambda i: i % self.gc.logging_freq == 0
    self.is_save_step = lambda i: (i + 1) % self.gc.save_freq == 0
    self.is_eval_step = lambda i: i % self.gc.eval_freq == 0


  @property
  def params(self):
    if self.optim_state is None:
      return None
    else:
      return self.optimizer.get_params(self.optim_state)


  def initialize(
      self,
      batch: Optional[FeatureDict] = None,
      load_format: Optional[str] = None,
      random_seed: Optional[int] = None):
    
    # create optimizer instance
    self.optimizer = Optimizer(self.optim_config)

    use_autoload = self.gc.start_step > 0
    if use_autoload:
      assert load_format is not None, \
          "must provide `load_format` to auto load models when assigning `start_step` > 0."
      self.autoload(self.gc.start_step, load_format)
    else:
      assert batch is not None, \
          "must provide a batch and a random seed to initialize model from scratch."
      if self.optim_state is not None:
        logging.warning("existed optimizer states are reinitialized.")
      if random_seed is not None:
        logging.warning("external random seed overrides the one in global config.")
      else:
        random_seed = self.gc.random_seed
        rng = jax.random.PRNGKey(random_seed)    # all ranks initialized equally.
      params = hk.data_structures.to_mutable_dict(self._init_fn(batch=batch, rng=rng))
      self.optim_state = self.optimizer.init_state(params)
    
    # define loss_fn
    def _loss_fn(params, batch, rng):
      # TODO: user external RNG
      _, loss = self._apply_fn(params=params, batch=batch, rng=rng)
      return loss 

    # define reduce_fn for mpi communication.
    if self.gc.use_mpi:
      # just-in-time imports
      from mpi4py import MPI
      import mpi4jax
      from jax.tree_util import tree_flatten, tree_unflatten
      def _mpi_reduce_value(value):
        value, _ = mpi4jax.allreduce(value, op=MPI.SUM, comm=self.mpi_comm)
        value /= self.world_size
        return value
      def _mpi_reduce_tree(tree):
        flat_tree, tree_struct = tree_flatten(tree)
        for i, val in enumerate(flat_tree):
          flat_tree[i] = _mpi_reduce_value(val)
        tree = tree_unflatten(tree_struct, flat_tree)
        return tree
    
    # define update_fn.  
    def _grad_fn(step, opt_state, batch, rng):
      loss, grads = jax.value_and_grad(_loss_fn)(
          self.optimizer.get_params(opt_state), batch, rng)
      grads = self.optimizer.clip_grads(grads)
      if self.gc.use_mpi:
        loss = _mpi_reduce_value(loss)
        grads = _mpi_reduce_tree(grads)
      return loss, grads
    
    # define eval_fn for validation.
    def _eval_fn(params, batch, rng):
      loss = _loss_fn(params, batch, rng)
      if self.gc.use_mpi:
        loss = _mpi_reduce_value(loss)
      return loss

    # define logit_fn to get logits.
    def _logit_fn(params, batch, rng):
      ret, loss = self._apply_fn(params=params, batch=batch, rng=rng)
      return ret, loss
    
    self._loss_fn = _loss_fn     # this is not re-jit as loss_fn is much of a wrapped apply_fn.
    self._eval_fn = _eval_fn
    self._logit_fn = _logit_fn
    self._grad_fn = jax.jit(_grad_fn)   # jit transformation of update_fn.
    
    # start ticking after initialization.
    self._tic = time.time()
  

  def autosave(self, step):
    # save ckpt in both npz and pkl formats.
    save_path_npz = os.path.join(self.gc.save_dir, self.auto_ckpt_name(step + 1, 'npz'))
    self.optimizer.save(self.optim_state, save_path_npz)
    save_path_pkl = os.path.join(self.gc.save_dir, self.auto_ckpt_name(step + 1, 'pkl'))
    self.optimizer.save(self.optim_state, save_path_pkl)
    # save loss curve
    train_curve_path = os.path.join(self.gc.save_dir, self.auto_curve_name(is_train=True))
    eval_curve_path = os.path.join(self.gc.save_dir, self.auto_curve_name(is_train=False))
    np.save(train_curve_path, np.asarray(self.train_losses))
    np.save(eval_curve_path, np.asarray(self.eval_losses))
    logging.info(f"model autosaved at step {step:05d} successfully.")


  def autoload(self, step, format='pkl'):
    # load ckpt
    load_path = os.path.join(self.gc.load_dir, self.auto_ckpt_name(step, format))
    self.optim_state = self.optimizer.load(load_path)
    # load loss curve
    train_curve_path = os.path.join(self.gc.save_dir, self.auto_curve_name(is_train=True))
    eval_curve_path = os.path.join(self.gc.save_dir, self.auto_curve_name(is_train=False))
    def load_loss_curve(loss_curve_path: str):
      try:
        return np.load(loss_curve_path).tolist()
      except:
        logging.warning(f"failed to load curve from {loss_curve_path}. reset loss curve.")
        return []
    self.train_losses = load_loss_curve(train_curve_path)
    self.eval_losses = load_loss_curve(eval_curve_path)
    logging.info(f"model autoloaded at step {step:05d} successfully.")


  def update(self, step, batch, rng, update, accumulation):
    def divide_pytree(pytree, div):
      return tree_map(lambda pt: pt / div, pytree)
    def add_pytrees(pytree1, pytree2):
      return tree_map(lambda pt1, pt2: pt1 + pt2, pytree1, pytree2)

    loss, grads = self._grad_fn(step, self.optim_state, batch, rng)
    if update:
      if self._grads is None:
        self._loss = loss/self.gc.gradient_accumulation
        self._grads = divide_pytree(grads, self.gc.gradient_accumulation)
      else:
        self._loss += loss/self.gc.gradient_accumulation
        self._grads = add_pytrees(self._grads, divide_pytree(grads, self.gc.gradient_accumulation))
      if not accumulation: #do update 
        self.optim_state = self.optimizer.opt_update(step, self._grads, self.optim_state)
        self._grads = None
      return self._loss
    return loss


  def _logging(self, step, loss):
    # print and record training stats at the step.
    toc = time.time()
    step_time = (toc - self._tic) / (
        1 if step == 0 else self.gc.logging_freq)
    self.train_losses.append((step, loss, step_time))
    logging.info(f"step: {step:05d}\ttrain_loss: {loss:3.4f}\tstep_time: {step_time:.2f}s")
    self._tic = time.time()


  def train_step(self, step, batch, rng, silent=True, update=True, accumulation=False):
    batch = cast_to_precision(batch, self.precision)
    loss = self.update(step, batch, rng, update, accumulation)
    if update:
      if not accumulation:
        add_scalar_log('loss/train', loss, step)
    else:
      add_scalar_log('loss/eval', loss, step)
    if not silent:
      if self.is_logging_step(step):
        self._logging(step, loss)
      if self.is_save_step(step):
        self.autosave(step)


  def eval_step(self, step, batch, rng, silent=True):
    # evaluation on the fly
    tmp_tic = time.time()
    loss = self._eval_fn(self.params, batch, rng)
    add_scalar_log('loss/eval', loss, step)
    eval_time = time.time() - tmp_tic
    if not silent:
      self.eval_losses.append((step, loss, eval_time))
      logging.info(f"step: {step:05d}\teval_loss:  {loss:3.4f}\teval_time: {eval_time:.2f}s")


  def build_distillation_step(self, step, batch, rng, silent=True):
    # save logits
    tmp_tic = time.time()
    value, loss = self._logit_fn(self.params, batch, rng)
    eval_time = time.time() - tmp_tic
    #if not silent:
    logging.info(f"step: {step:05d}\teval_loss:  {loss:3.4f}\teval_time: {eval_time:.2f}s")
    return value, loss




