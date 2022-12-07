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
from multiprocessing import Process, Queue
import numpy as np
import os
import gzip, pickle
from copy import deepcopy
from os.path import exists

from deepfold.common.residue_constants import sequence_to_onehot
from deepfold.model.features import FeatureDict
from deepfold.model.features import np_example_to_features as process_features
from deepfold.train import utils
from deepfold.train.label_pipeline import process_labels

SAVE_MMCIF_PKL = False

FEATNAME_DICT = set(['aatype', 'residue_index', 'seq_length', 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_sum_probs', 'is_distillation', 'seq_mask', 'msa_mask', 'msa_row_mask', 'random_crop_to_size_seed', 'template_mask', 'template_pseudo_beta', 'template_pseudo_beta_mask', 'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'extra_msa', 'extra_msa_mask', 'extra_msa_row_mask', 'bert_mask', 'true_msa', 'extra_has_deletion', 'extra_deletion_value', 'msa_feat', 'target_feat','ss_type','ss_mask','PHI_PSI','PHI_CHI1','PSI_CHI1','CHI1_CHI2'])

def dict_str(dic, dep):
    ret = ""
    for k, v in dic.items():
        ret += f'\n{"  "*dep}'
        if type(v).__name__ in ['Vecs','Rigids','Rots']:
            v = v._asdict()
        if type(v) is dict:
            ret += f'{k}'
            ret += dict_str(v, dep+1)
        elif k=='name':
            ret += f'{k}='
            if len(v.shape) == 2:
              ret += "".join(chr(i) for i in v[0])
            else:
              for vi in v:
                ret += "".join(chr(i) for i in vi[0])
                ret += ", "
        elif v.shape==(1,):
            ret += f'{k}={v[0]}'
        elif v.shape==():
            ret += f'{k}={v:.4f}'
        else:
            ret += f'{k}={v.shape}'
    return ret
  
def cast_to_precision(batch, precision):
  # the input batch is asserted of precision fp32.
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
  def __init__(self,
               model_config,      # model config.
               data_config,       # data config mainly including paths.
               build_distillation=False
               ):      
    # copy config
    self.mc = model_config
    self.dc = data_config

    # distillation build setting
    self.build_distillation = build_distillation
    if self.build_distillation:
      self.build_list = DataSystem.get_build_list_from_json(self.dc.build_list)
      self.num_build = len(self.build_list)
      self.cur_build = 0

    # get crop size
    self.crop_size = self.mc.data.eval.crop_size

    # get sample_weights
    if self.dc.sample_weights is None:
      sample_weights = DataSystem.get_sample_weights_from_dir(self.dc.features_dir)
    else:   # use all entries under
      sample_weights = DataSystem.get_sample_weights_from_json(self.dc.sample_weights)

    self.prot_keys = list(sample_weights.keys())
    # unify the sample weights as sample probability
    sum_weights = sum(sample_weights.values())
    self.sample_prob = [v / sum_weights for v in sample_weights.values()]
    self.num_prot = len(self.prot_keys)
    self.check_completeness()


  def check_completeness(self):
    # check that every protein has mmcif as labels.
    #self.pdb_list = DataSystem.get_pdb_list_from_dir(self.dc.mmcif_dir)
    for prot_name in self.prot_keys:
      pdb_id = prot_name.split('_')[0]
      assert not os.path.exists(os.path.join(self.dc.mmcif_dir, f"/{pdb_id[1:3]}/{pdb_id}.cif.gz")), \
          "%s doesn't have the corresponding mmcif file in %s." % (prot_name, self.dc.mmcif_dir)
    logging.debug("checking for data completeness successful.")


  def load(self, prot_name: str):
    feature_path = os.path.join(self.dc.features_dir, prot_name+'/features.pkl')
    if not os.path.exists(feature_path):
         feature_path=os.path.join(self.dc.features_dir, f'{prot_name[1:3]}/{prot_name}/features.pkl')
    pdb_id = prot_name[:4]
    chain_id = prot_name.split('_')[-1]
    dssp_path = os.path.join(self.dc.dssp_dir,f'{prot_name[1:3]}/{pdb_id + chain_id}/dssp.pkl')
    raw_features = utils.load_features(feature_path)
    dssp_features = utils.load_features(dssp_path)
    prot_info = prot_name.split('_')    # assert naming styles are in `101m_A` or `101m_1_A`
    pdb_id, chain_id = prot_info[0], prot_info[-1]

    # literature process
    with gzip.open('literature/binned_ramachan.pkl', 'rb') as f:
      binned_ramachan = pickle.load(f)
      for name in ['PHI_PSI', 'PHI_CHI1', 'PSI_CHI1', 'CHI1_CHI2']:
        raw_features[name] = binned_ramachan[name]['TOTAL']
    
    # secondary structure
    #raw_features['ss_label'] = dssp_features['ss_label']
    raw_features['ss_type'] = jnp.asarray(dssp_features['ss_type'])
    ss_mask = jnp.ones(dssp_features['ss_type'].shape[0])
    ss_mask = jnp.where(jnp.sum(dssp_features['ss_type'],axis=-1)==0, 0, ss_mask)
    raw_features['ss_mask'] = ss_mask

    # pdb label checkpoint
    raw_labels_path=os.path.join(self.dc.mmcif_dir, f'{pdb_id[1:3]}/{pdb_id}_{chain_id}.label.gz')
    if os.path.exists(raw_labels_path):
      with gzip.open(raw_labels_path, 'rb') as f:
        raw_labels = pickle.load(f)
    else:
      cif_path=os.path.join(self.dc.mmcif_dir, pdb_id+'.cif')
      if not os.path.exists(cif_path):
          cif_path=os.path.join(self.dc.mmcif_dir, f'{pdb_id[1:3]}/{pdb_id}.cif.gz')

      raw_labels = utils.load_labels(
          #cif_path=os.path.join(self.dc.mmcif_dir, pdb_id+'.cif'),
          cif_path=cif_path,
          pdb_id=pdb_id,
          chain_id=chain_id)

      if SAVE_MMCIF_PKL:
        with gzip.open(raw_labels_path, 'wb') as f:
          pickle.dump(raw_labels, f)

    return raw_features, raw_labels


  def preprocess(
      self,
      rng,
      raw_features: FeatureDict,
      raw_labels: FeatureDict,
      crop_start: None) -> FeatureDict:
    rng, crop_seed = utils.split_np_random_seed(rng)

    #    logging.error(dict_str(raw_features ,0))

    raw_features, raw_labels, crop_start = utils.crop_and_pad(
        raw_features, raw_labels,
        crop_size=self.crop_size,
        pad_for_shorter_seq=True,
        random_seed=crop_seed,
        crop_start=crop_start)
    rng, feat_seed = utils.split_np_random_seed(rng)
    processed_features = process_features(
        raw_features,
        config=self.mc,
        random_seed=feat_seed)
    with jax.disable_jit():           # using jit here is experimentally slower
      processed_labels = process_labels(raw_labels)
    batch = {**processed_features, **processed_labels}
    if crop_start is None:
        crop_start=0
    batch['crop_start'] = np.array([crop_start])
    for name in ['ss_type', 'ss_mask', 'PHI_PSI', 'PHI_CHI1', 'PSI_CHI1', 'CHI1_CHI2']:
      batch[name] = np.array([raw_features[name]])
    return rng, batch


  def sample(
      self,
      rng,
      batch_size = None):
    """
    pick a (batch of) protein(s) randomly and generate rng(s) for processing.
    if batch_size is None, return a pair of result; otherwise return a list of pairs.
    """
    rng, seed = utils.split_np_random_seed(rng)
    np.random.seed(seed)
    if batch_size is None:
      prot_idx = np.random.choice(self.num_prot, p=self.sample_prob)
      return rng, prot_idx
    else:         # this code is not used.
      prot_idxs = np.random.choice(
          self.num_prot,
          size=batch_size,
          replace=(batch_size > self.num_prot),
          p=self.sample_prob)
      rngs = list(jrand.split(rng, batch_size))
      return list(zip(rngs, prot_idxs))

  def zero_template_add(self, feat):
    seq_len = feat['aatype'].shape[0]
    feat['template_aatype']=np.zeros((1,seq_len,22))
    feat['template_all_atom_masks']=np.zeros((1,seq_len,37))
    feat['template_all_atom_positions']=np.zeros((1,seq_len,37,3))
    feat['template_domain_names']=np.zeros((1,))
    feat['template_sequence']=np.zeros((1,))
    feat['template_sum_probs']=np.zeros((1,1))
    return feat
  
  def get_batch(
      self,
      prot_idx,
      rng):
    if self.build_distillation:
        prot_name, crop_start = self.build_list[prot_idx % self.num_build]
    else:
        prot_name = self.prot_keys[prot_idx % self.num_prot]
        crop_start = None
    logging.debug(f"loading protein #{prot_idx:06d}: {prot_name}...")
    #logging.error(f"loading protein {prot_name}...") ####
    raw_features, raw_labels = self.load(prot_name)
    #logging.error(f"load end") ####
    #cy3: insert empty template
    if raw_features["template_aatype"].shape == (0,):
      raw_features = self.zero_template_add(raw_features)

    resolution = raw_labels.pop('resolution')
    rng, batch = self.preprocess(rng, raw_features, raw_labels, crop_start=crop_start)
    batch['name'] = np.array([list(prot_name.encode('ascii'))])
    batch['resolution'] = resolution
    rng, batch_rng = jrand.split(rng, 2)
    #logging.error(f"preprocess end") ####
    return rng, batch_rng, batch
  

  def random_recycling(
      self,
      step,
      batch):
    """
    generate the number of recycling iterations for a given step and add it to the batch.
    this method is specifically set here to make sure the result is equal among workers at each step.
    """
    rng = jrand.PRNGKey(step)
    num_iter_recycling = jrand.randint(rng, [1], 0, self.mc.model.num_recycle + 1)
    batch['num_iter_recycling'] = num_iter_recycling
    return batch

  
  def batch_gen(self, rng):
    with jax.disable_jit():
      if self.build_distillation:
          while True:
            prot_idx = self.cur_build
            self.cur_build += 1
            rng, batch_rng, batch = self.get_batch(prot_idx, rng)
            yield batch_rng, batch
      else:
          while True:
            rng, prot_idx = self.sample(rng, None)
            rng, batch_rng, batch = self.get_batch(prot_idx, rng)
            yield batch_rng, batch


  @staticmethod
  def get_sample_weights_from_dir(features_dir):
    sample_weights = {
      os.path.basename(p): 1
          for p in glob.glob(features_dir + "/*") if os.path.isdir(p)
    }
    assert len(list(sample_weights.keys())) > 0, \
        "no sub-directories under given feature directory %s." % (features_dir)
    return sample_weights

  @staticmethod
  def get_build_list_from_json(json_path):
    try:
      build_list = json.load(open(json_path, 'r'))
    except:
      raise ValueError("failed to load sample weights from json file %s." % json_path)
    
    ret = []
    if isinstance(build_list, dict):
      for prot_name in build_list:
        ret += [(prot_name, crop_start) for crop_start in build_list[prot_name]]
    return ret

  @staticmethod
  def get_sample_weights_from_json(json_path):
    try:
      sample_weights = json.load(open(json_path, 'r'))
    except:
      raise ValueError("failed to load sample weights from json file %s." % json_path)
    if isinstance(sample_weights, list):
      sample_weights = {k: 1 for k in sample_weights}
    return sample_weights


  @staticmethod
  def get_pdb_list_from_dir(mmcif_dir):
    pdb_list = [
        os.path.basename(mmcif).replace('.cif','')
            for mmcif in glob.glob(mmcif_dir + "/*.cif")]
    pdb_list.append([ os.path.basename(mmcif).replace('.cif.gz','')
      for mmcif in glob.glob(os.path.join(mmcif_dir, "/**/*.cif.gz"), recursive=True)])
    return pdb_list


class GetBatchProcess(Process):
  """
  a multiprocessing worker to conduct data loading.
  remark: make sure no jax call is used before this worker starts,
          or the XLA-in-fork issue could arise. (notably, there are 
          tensorflow calls in `DataSystem.preprocess()`. )
  """
  def __init__(
      self,
      queue: Queue,
      data: DataSystem,
      num_batches: int,           # number of batches to generate
      is_training: bool = True,   # if true, random recycling is used.
      random_seed: int = 0,
      mpi_rank: int = 0):
    Process.__init__(self)
    self.queue = queue
    self.data = data
    self.num_batches = num_batches
    self.is_training = is_training
    self.random_seed = random_seed
    self.mpi_rank = mpi_rank

  def run(self):
    with jax.disable_jit():
      rng = jrand.PRNGKey(self.random_seed)
      rng = jrand.fold_in(rng, self.mpi_rank)
    batch_gen = self.data.batch_gen(rng)
    for step in range(self.num_batches):
      batch_rng, batch = next(batch_gen)
      if self.is_training:
        batch = self.data.random_recycling(step, batch)
      self.queue.put((batch_rng, batch))
      logging.debug(f"write queue item {step}. current qsize = {self.queue.qsize()}.")
    logging.debug("get batch process finished.")
    return
