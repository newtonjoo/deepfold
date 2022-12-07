# dataloader test 

import unittest
from unittest import TestCase
import logging
from ml_collections import ConfigDict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
from deepfold.model.config import model_config
data_config = ConfigDict({
            # Directory to store features (features.pkl files)
            'features_dir': "example_data/features",
            # Directory to store labels (.mmcif files)
            'mmcif_dir': "example_data/mmcif",
            # Directory to store secondary_structure (.pkl file)
            'dssp_dir' : "example_data/dssp",
            # Json file that specifies sampling weights of each sample.
            'sample_weights': "example_data/sample_weights.json"
        })
model_config = model_config("model1")
rng = jax.random.PRNGKey(42)

from deepfold.train.data_system import DataSystem
#from tests.test_dataloader import data_config
#data_loader = DataSystem(model_config, data_config)
#batch_gen = data_loader.batch_gen(rng)

NUM_RES = 256
NUM_MSA_SEQ = 508
#NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
#NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES

# batch keys and shape
assert_batch_dict = {
  "aatype":(4, NUM_RES),
  "residue_index":(4, NUM_RES),
  "seq_length":(4,),
  "msa":(4, NUM_MSA_SEQ, NUM_RES),
  "template_aatype":(4, 4, NUM_RES),
  "template_all_atom_masks":(4, 4, NUM_RES, 37),
  "template_all_atom_positions":(4, 4, NUM_RES, 37, 3),
  "template_sum_probs":(4, 4, 1),
  "is_distillation":(4,),
  "seq_mask":(4, NUM_RES),
  "msa_mask":(4, NUM_MSA_SEQ, NUM_RES),
  "msa_row_mask":(4, NUM_MSA_SEQ),
  "random_crop_to_size_seed":(4, 2),
  "template_mask":(4, 4),
  "template_pseudo_beta":(4, 4, NUM_RES, 3),
  "template_pseudo_beta_mask":(4, 4, NUM_RES),
  "atom14_atom_exists":(4, NUM_RES, 14),
  "residx_atom14_to_atom37":(4, NUM_RES, 14),
  "residx_atom37_to_atom14":(4, NUM_RES, 37),
  "atom37_atom_exists":(4, NUM_RES, 37),
  "extra_msa":(4, 1024, NUM_RES),
  "extra_msa_mask":(4, 1024, NUM_RES),
  "extra_msa_row_mask":(4, 1024),
  "bert_mask":(4, NUM_MSA_SEQ, NUM_RES),
  "true_msa":(4, NUM_MSA_SEQ, NUM_RES),
  "extra_has_deletion":(4, 1024, NUM_RES),
  "extra_deletion_value":(4, 1024, NUM_RES),
  "msa_feat":(4, NUM_MSA_SEQ, NUM_RES, 49),
  "target_feat":(4, NUM_RES, 22),
  "aatype_index":(1, NUM_RES),
  "all_atom_positions":(1, NUM_RES, 37, 3),
  "all_atom_mask":(1, NUM_RES, 37),
  "sequence":(1,),
  "pseudo_beta":(1, NUM_RES, 3),
  "pseudo_beta_mask":(1, NUM_RES),
  "atom14_gt_exists":(1, NUM_RES, 14),
  "atom14_gt_positions":(1, NUM_RES, 14, 3),
  "atom14_alt_gt_positions":(1, NUM_RES, 14, 3),
  "atom14_alt_gt_exists":(1, NUM_RES, 14),
  "atom14_atom_is_ambiguous":(1, NUM_RES, 14),
  "backbone_affine_tensor":(1, NUM_RES, 7),
  "backbone_affine_mask":(1, NUM_RES),
  "rigidgroups_gt_frames":(1, NUM_RES, 8, 12),
  "rigidgroups_gt_exists":(1, NUM_RES, 8),
  "rigidgroups_group_exists":(1, NUM_RES, 8),
  "rigidgroups_group_is_ambiguous":(1, NUM_RES, 8),
  "rigidgroups_alt_gt_frames":(1, NUM_RES, 8, 12),
  "chi_angles_sin_cos":(1, NUM_RES, 4, 2),
  "chi_mask":(1, NUM_RES, 4),
  "crop_start":(1,),
  "ss_type":(1, NUM_RES, 8),
  "ss_mask":(1, NUM_RES),
  "PHI_PSI":(1, 36, 36),
  "PHI_CHI1":(1, 36, 36),
  "PSI_CHI1":(1, 36, 36),
  "CHI1_CHI2":(1, 36, 36),
  "name":(1, 8),
  "resolution":(1,),
}

class TestDataLoader(TestCase):
  def __init__(self, *args):
    super().__init__(*args)

    self.data_loader = DataSystem(model_config, data_config)

  def test_dataloader(self):
    batch_gen = self.data_loader.batch_gen(rng)
    batch_rng, batch = next(batch_gen)

    for k,v in assert_batch_dict.items():
      #log.info(f"DataloaderTest: {batch[k].shape}")
      self.assertEqual(batch[k].shape, v, msg=f"{k} has wrong shape {batch[k].shape} != {v}")


