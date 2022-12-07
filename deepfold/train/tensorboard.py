from tensorboardX import SummaryWriter
from jax.experimental.host_callback import id_tap
import jax.numpy as jnp
from absl import logging
import os
summary_writer = None
last_step = 0

def init_writer():
  global summary_writer
  logging.info('summary_writer init')
  if 'custom_config' in os.environ:
    summary_writer = SummaryWriter(f"runs/{os.environ['custom_config']}")
  else:
    summary_writer = SummaryWriter()

def add_scalar_log(name, value, step=None):
  if summary_writer is not None:
    global last_step
    if step is None:
      step = last_step
      if step == 0:
        return
    else:
      last_step = step
    #logging.info(f'log: {name}, {value}, {step}')
    summary_writer.add_scalar(name, value, step)

def add_jax_log(arg, transforms):
  name, val = arg
  add_scalar_log("".join(chr(i) for i in name),val)

def jax_log(name, value):
  id_tap(add_jax_log,(jnp.array(list(name.encode('ascii'))),value))
