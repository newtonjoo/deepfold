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

"""Data system used to load training datasets."""

from absl import logging
import glob
import jax
import jax.numpy as jnp
import jax.random as jrand
import json
from multiprocessing import Process, Queue, Pool
import numpy as np
import os
import lzma
import gzip
import pickle
from deepfold.train import utils

FEATNAME_DICT = set(['aatype', 'residue_index', 'seq_length', 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_sum_probs', 'is_distillation', 'seq_mask', 'msa_mask', 'msa_row_mask', 'random_crop_to_size_seed', 'template_mask', 'template_pseudo_beta', 'template_pseudo_beta_mask', 'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'extra_msa', 'extra_msa_mask', 'extra_msa_row_mask', 'bert_mask', 'true_msa', 'extra_has_deletion', 'extra_deletion_value', 'msa_feat', 'target_feat','ss_type','ss_mask','PHI_PSI','PHI_CHI1','PSI_CHI1','CHI1_CHI2'])

# logits
logit_features =[('teacher_distogram_logits',       ['distogram', 'logits']),
                 ('teacher_exp_resolved_logits',    ['experimentally_resolved', 'logits']),
                 ('teacher_masked_msa_logits',      ['masked_msa', 'logits']),
                 ('teacher_predicted_lddt_logits',  ['predicted_lddt', 'logits'])
                ]
# represetations
repr_features = [('teacher_input_pair',         ['representations', 'input_pair']),
                 ('teacher_msa',                ['representations', 'msa']),
                 ('teacher_pair',               ['representations', 'pair']),
                 ('teacher_single',             ['representations', 'single']),
                 ('teacher_ipa',                ['representations', 'ipa']),
                 ('teacher_backbone',           ['representations', 'backbone']),
                 ('teacher_structure_module',   ['representations', 'structure_module'])
                ]

def key_exist(dic, keys):
  for key in keys:
    if key not in dic:
      return False
    dic = dic[key]
  return True

def nested_get(dic, keys):
  for key in keys:
      dic = dic[key]
  return dic

def cast_to_precision(batch, precision):
  # the input batch is asserted of precision fp32.I
  if precision == 'bf16':
    dtype = jnp.bfloat16
  elif precision == 'fp16':
    dtype = jnp.float16
  else:   # assert fp32 specified
    return batch
  for key in batch:
    # skip int type
    if batch[key].dtype in [np.int32, np.int64, jnp.int32, jnp.int64]:
      continue
    if 'feat' in key or 'mask' in key or key in FEATNAME_DICT:
      batch[key] = jnp.asarray(batch[key], dtype=dtype)
  return batch

class DataSystem:
  """
  multiprocess data system
  """
  def __init__(self, data_list_path, dssp_path=None):      
    try:
      self.data_list = json.load(open(data_list_path, 'r'))
    except:
      raise ValueError("failed to load list from json file %s." % data_list_path)
    self.dssp_path = dssp_path
    self.num_data = len(self.data_list)

  def __len__(self):
      return self.num_data

  @staticmethod
  def check_batch(batch):
    for key in FEATNAME_DICT:
        if key not in batch:
            raise ValueError(f"{key} is not in batch")

  @staticmethod
  def process_batch(pre_batch):
    batch = pre_batch['batch']
    teacher = pre_batch['value']


    for feature_name, keys in logit_features:
      if key_exist(teacher, keys):
        batch[feature_name] = np.expand_dims(nested_get(teacher, keys), axis=0)

    for feature_name, keys in repr_features:
      if key_exist(teacher, keys):
        batch[feature_name] = np.expand_dims(nested_get(teacher, keys), axis=0)

    # batch name crop
    batch["name"] = batch["name"][..., :8]

    return batch
  
  def __getitem__(self, idx):
    filename = self.data_list[idx]
    datafile = lzma.LZMAFile(filename, 'r')
    pre_batch = np.load(datafile, allow_pickle=True)
    batch = DataSystem.process_batch(pre_batch)

    # secondary_structure process
    if self.dssp_path:
      basename = os.path.basename(filename)
      pdb_id, model_id, chain_id, crop_start = basename.split('.')[0].split('_')
      dssp_path = os.path.join(self.dssp_path,f'{pdb_id[1:3]}/{pdb_id + chain_id}/dssp_{crop_start}.pkl')
      dssp_features = utils.load_features(dssp_path)
      batch['ss_type'] = np.expand_dims(
          np.array(dssp_features['ss_type'],dtype=np.int32),axis=0)
      batch['ss_mask'] = np.expand_dims(
          np.array(dssp_features['ss_mask'],dtype=np.float32),axis=0)

    # literature process
    with gzip.open('literature/binned_ramachan.pkl', 'rb') as f:
      binned_ramachan = pickle.load(f)
      for name in ['PHI_PSI', 'PHI_CHI1', 'PSI_CHI1', 'CHI1_CHI2']:
        batch[name] = np.expand_dims(binned_ramachan[name]['TOTAL'],axis=0)

    DataSystem.check_batch(batch)
    return batch
  
  def queue_writer(self, queue, num_worker=None, order=None):
    np.random.seed()
    logging.info(f"writer {os.getpid()} start.")
    idx = order
    while True:
      if order is None:
        idx = np.random.randint(self.num_data)
      else:
        idx += num_worker
        if idx >= self.num_data:
          return
      try:
        batch =self[idx]
        queue.put(batch)
      except:
        logging.error(f'error occured while loading {self.data_list[idx]}')
      logging.debug(f"{os.getpid()}: current qsize = {queue.qsize()}.")

