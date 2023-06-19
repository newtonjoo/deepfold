# Copyright 2022 Deepfold Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model config."""

import copy
from deepfold.model.tf import shape_placeholders
from ml_collections import ConfigDict


NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES


def model_config(
    name: str,
    is_training: bool = False,
    use_ptm: bool = False) -> ConfigDict:
  """Get the ConfigDict of a Deepfold model."""

  if name not in CONFIG_DIFFS:
    raise ValueError(f'Invalid model name {name}.')
  cfg = copy.deepcopy(CONFIG)
  cfg.update_from_flattened_dict(CONFIG_DIFFS[name])
  if is_training:
    # This must be done for any attempt of training.
    cfg.update_from_flattened_dict(CONFIG_FOR_TRAIN)
  if use_ptm:
    cfg.update_from_flattened_dict(CONFIG_FOR_PTM)
  return cfg


CONFIG_DIFFS = {
  # Users may create customized configs by adding entries in this dict
  # and referring to them via the key name.
  # The configuration we used to train Deepfold:
  'model1': {
    #skip evoformer module
    'data.eval.crop_size': 256,
    'data.common.reduce_msa_clusters_by_max_templates': True,
    'data.common.use_templates': True,
    'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
    'model.embeddings_and_evoformer.template.enabled': True,
    #'model.embeddings_and_evoformer.skip_module': True,
  },
  'model2' : {
    'data.common.reduce_msa_clusters_by_max_templates': True,
    'data.common.use_templates': True,
    'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
    'model.embeddings_and_evoformer.template.enabled': True,
    #'model.embeddings_and_evoformer.skip_module': True,
    'model.heads.structure_module.structural_violation_loss_weight': 0.1,
    'model.heads.predicted_sc_confidence.weight': 0.01,
    'model.heads.structure_module.chi_dependent' : True,
    'model.heads.structure_module.bb_weight': 1.0,
    'model.heads.structure_module.chi_weight': 1.0,
    'model.heads.structure_module.pair_weight': 0.0,
    'model.heads.structure_module.pair_epsilon': 0.0,
    'model.heads.structure_module.phipsi_lit_weight': 0.1,
    'model.heads.structure_module.phichi1_lit_weight': 0.1,
    'model.heads.structure_module.psichi1_lit_weight': 0.1,
    'model.heads.structure_module.secondary.weight_frac': 0.0,
  },
  'model3' : {
    'data.common.reduce_msa_clusters_by_max_templates': True,
    'data.common.use_templates': True,
    'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
    'model.embeddings_and_evoformer.template.enabled': True,
    #'model.embeddings_and_evoformer.skip_module': True,
    'model.heads.structure_module.structural_violation_loss_weight': 0.1,
    'model.heads.predicted_sc_confidence.weight': 0.00,
    'model.heads.structure_module.loss_type' : 'root',
    'model.heads.structure_module.chi_dependent' : True,
    'model.heads.structure_module.bb_weight': 0.0,
    'model.heads.structure_module.chi_weight': 1.0,
    'model.heads.structure_module.chi1_weight': 1.0,
    'model.heads.structure_module.chi2_weight': 1.0,
    'model.heads.structure_module.chi3_weight': 1.0,
    'model.heads.structure_module.chi4_weight': 1.0,
    'model.heads.structure_module.pair_weight': 0.0,
    'model.heads.structure_module.pair_epsilon': 0.0,
    'model.heads.structure_module.phipsi_lit_weight': 0.00,
    'model.heads.structure_module.phichi1_lit_weight': 0.00,
    'model.heads.structure_module.psichi1_lit_weight': 0.00,
    'model.heads.structure_module.secondary.weight_frac': 0.0,
  },
  'model4': {
    'data.common.reduce_msa_clusters_by_max_templates': True,
    'data.common.use_templates': True,
    'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
    'model.embeddings_and_evoformer.template.enabled': True,

    # deactivate sc_conf (jinjin)
    'model.heads.predicted_sc_confidence.weight': 0.1,

    # deactivate sidechain loss (jsg)
    'model.heads.structure_module.loss_type' : 'root',
    'model.heads.structure_module.chi_dependent' : False,
    'model.heads.structure_module.bb_weight': 1.5,
    'model.heads.structure_module.chi_weight': 1.5,
    'model.heads.structure_module.phipsi_lit_weight': 0.0,
    'model.heads.structure_module.phichi1_lit_weight': 0.0,
    'model.heads.structure_module.psichi1_lit_weight': 0.0,
    # not for use
    'model.heads.structure_module.pair_weight': 0.0,
    'model.heads.structure_module.pair_epsilon': 0.0,

    # deactivate secondary loss (yujin)
    'model.heads.structure_module.secondary.weight_frac': 0.05,

    # deactivate weighted FAPE loss (jhw)
    'model.heads.structure_module.sidechain.weight_param_alpha': 2.0,
    'model.heads.structure_module.sidechain.weight_param_h_offset': 12.0,
    'model.heads.structure_module.sidechain.weight_param_v_offset': 1.5,

    'model.heads.structure_module.num_layer': 8, #8
  },
  'model5': {
    'data.common.reduce_msa_clusters_by_max_templates': False,
    'data.common.use_templates': False,
    'model.embeddings_and_evoformer.template.embed_torsion_angles': False,
    'model.embeddings_and_evoformer.template.enabled': False,

    # deactivate sc_conf (jinjin)
    'model.heads.predicted_sc_confidence.weight': 0.1,

    # deactivate sidechain loss (jsg)
    'model.heads.structure_module.loss_type' : 'root',
    'model.heads.structure_module.chi_dependent' : False,
    'model.heads.structure_module.bb_weight': 1.5,
    'model.heads.structure_module.chi_weight': 1.5,
    'model.heads.structure_module.phipsi_lit_weight': 0.0,
    'model.heads.structure_module.phichi1_lit_weight': 0.0,
    'model.heads.structure_module.psichi1_lit_weight': 0.0,
    # not for use
    'model.heads.structure_module.pair_weight': 0.0,
    'model.heads.structure_module.pair_epsilon': 0.0,

    # deactivate secondary loss (yujin)
    'model.heads.structure_module.secondary.weight_frac': 0.0,

    # deactivate weighted FAPE loss (jhw)
    'model.heads.structure_module.sidechain.weight_param_alpha': 2.0,
    'model.heads.structure_module.sidechain.weight_param_h_offset': 12.0,
    'model.heads.structure_module.sidechain.weight_param_v_offset': 1.5,

    'model.heads.structure_module.num_layer': 8, #8
  },
  # A demo configuration for debugging:
  'demo': {
    'data.common.max_extra_msa': 128,
    'data.common.num_recycle': 1,
    'data.eval.crop_size': 128,
    'data.eval.max_msa_cluster': 16,
    'model.embeddings_and_evoformer.evoformer_num_block': 4,
    'model.embeddings_and_evoformer.extra_msa_channel': 16,
    'model.embeddings_and_evoformer.msa_channel': 128,
    'model.embeddings_and_evoformer.pair_channel': 64,
    'model.embeddings_and_evoformer.seq_channel': 192,
    'model.heads.structure_module.num_layer': 2,
    'model.heads.structure_module.num_channel': 192,
    'model.heads.structure_module.num_head': 6,
    'model.heads.predicted_lddt.num_channels': 64,
    'model.num_recycle': 1,
    'model.heads.experimentally_resolved.weight': 0.0,
    'model.heads.structure_module.structural_violation_loss_weight': 0.0
  },
}

CONFIG_FOR_TRAIN = {
  'data.eval.subsample_templates': True,
  'model.global_config.use_remat': True
}

CONFIG_FOR_PTM = {
  # If updated into the ConfigDict, the model will be trained with an 
  # additional predicted_aligned_error head that can produce predicted 
  # TM-score (pTM) and predicted aligned errors. 
  'model.heads.predicted_aligned_error.weight': 0.1
}

# default model configurations in Alphafold 2 repo, unaltered.
CONFIG = ConfigDict({
  'data': {
    'common': {
      'masked_msa': {
        'profile_prob': 0.1,
        'same_prob': 0.1,
        'uniform_prob': 0.1
      },
      'max_extra_msa': 1024,
      'msa_cluster_features': True,
      'num_recycle': 10,
      'reduce_msa_clusters_by_max_templates': False,
      'resample_msa_in_recycling': True,
      'template_features': [
        'template_all_atom_positions', 'template_sum_probs',
        'template_aatype', 'template_all_atom_masks',
        'template_domain_names'
      ],
      'unsupervised_features': [
        'aatype', 'residue_index', 'sequence', 'msa', 'domain_name',
        'num_alignments', 'seq_length', 'between_segment_residues',
        'deletion_matrix'
      ],
      'use_templates': False,
    },
    'eval': {
      'feat': {
        'aatype': [NUM_RES],
        'all_atom_mask': [NUM_RES, None],
        'all_atom_positions': [NUM_RES, None, None],
        'alt_chi_angles': [NUM_RES, None],
        'atom14_alt_gt_exists': [NUM_RES, None],
        'atom14_alt_gt_positions': [NUM_RES, None, None],
        'atom14_atom_exists': [NUM_RES, None],
        'atom14_atom_is_ambiguous': [NUM_RES, None],
        'atom14_gt_exists': [NUM_RES, None],
        'atom14_gt_positions': [NUM_RES, None, None],
        'atom37_atom_exists': [NUM_RES, None],
        'backbone_affine_mask': [NUM_RES],
        'backbone_affine_tensor': [NUM_RES, None],
        'bert_mask': [NUM_MSA_SEQ, NUM_RES],
        'chi_angles': [NUM_RES, None],
        'chi_mask': [NUM_RES, None],
        'extra_deletion_value': [NUM_EXTRA_SEQ, NUM_RES],
        'extra_has_deletion': [NUM_EXTRA_SEQ, NUM_RES],
        'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
        'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
        'extra_msa_row_mask': [NUM_EXTRA_SEQ],
        'is_distillation': [],
        'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
        'msa_mask': [NUM_MSA_SEQ, NUM_RES],
        'msa_row_mask': [NUM_MSA_SEQ],
        'pseudo_beta': [NUM_RES, None],
        'pseudo_beta_mask': [NUM_RES],
        'random_crop_to_size_seed': [None],
        'residue_index': [NUM_RES],
        'residx_atom14_to_atom37': [NUM_RES, None],
        'residx_atom37_to_atom14': [NUM_RES, None],
        'resolution': [],
        'rigidgroups_alt_gt_frames': [NUM_RES, None, None],
        'rigidgroups_group_exists': [NUM_RES, None],
        'rigidgroups_group_is_ambiguous': [NUM_RES, None],
        'rigidgroups_gt_exists': [NUM_RES, None],
        'rigidgroups_gt_frames': [NUM_RES, None, None],
        'seq_length': [],
        'seq_mask': [NUM_RES],
        'target_feat': [NUM_RES, None],
        'template_aatype': [NUM_TEMPLATES, NUM_RES],
        'template_all_atom_masks': [NUM_TEMPLATES, NUM_RES, None],
        'template_all_atom_positions': [
          NUM_TEMPLATES, NUM_RES, None, None],
        'template_backbone_affine_mask': [NUM_TEMPLATES, NUM_RES],
        'template_backbone_affine_tensor': [
          NUM_TEMPLATES, NUM_RES, None],
        'template_mask': [NUM_TEMPLATES],
        'template_pseudo_beta': [NUM_TEMPLATES, NUM_RES, None],
        'template_pseudo_beta_mask': [NUM_TEMPLATES, NUM_RES],
        'template_sum_probs': [NUM_TEMPLATES, None],
        'true_msa': [NUM_MSA_SEQ, NUM_RES],
        'msa' : [NUM_MSA_SEQ,NUM_RES],
        'categorical_bert':[NUM_MSA_SEQ,NUM_RES],
        'ss_mask' : [NUM_RES],
        'ss_type' : [NUM_RES, None],
        #'PHI_PSI' : [None, None],
        #'PHI_CHI1' : [None, None],
        #'PSI_CHI1' : [None, None],
        #'CHI1_CHI2' : [None, None],
      },
      'fixed_size': True,
      'crop_size': 256,              # No cropping is used in inference.
      'subsample_templates': False,  # We want top templates.
      'masked_msa_replace_fraction': 0.15,
      'max_msa_clusters': 512,
      'max_templates': 4,
      'num_ensemble': 1,
    },
  },
  'model': {
    'embeddings_and_evoformer': {
      'stop_gradient': False,
      'skip_module': False,
      'evoformer_num_block': 48,
      'evoformer': {
        'msa_row_attention_with_pair_bias': {
          'dropout_rate': 0.15,
          'gating': True,
          'num_head': 8,
          'orientation': 'per_row',
          'shared_dropout': True
        },
        'msa_column_attention': {
          'dropout_rate': 0.0,
          'gating': True,
          'num_head': 8,
          'orientation': 'per_column',
          'shared_dropout': True
        },
        'msa_transition': {
          'dropout_rate': 0.0,
          'num_intermediate_factor': 4,
          'orientation': 'per_row',
          'shared_dropout': True
        },
        'outer_product_mean': {
          'chunk_size': 128,
          'dropout_rate': 0.0,
          'num_outer_channel': 32,
          'orientation': 'per_row',
          'shared_dropout': True
        },
        'triangle_attention_starting_node': {
          'dropout_rate': 0.25,
          'gating': True,
          'num_head': 4,
          'orientation': 'per_row',
          'shared_dropout': True
        },
        'triangle_attention_ending_node': {
          'dropout_rate': 0.25,
          'gating': True,
          'num_head': 4,
          'orientation': 'per_column',
          'shared_dropout': True
        },
        'triangle_multiplication_outgoing': {
          'dropout_rate': 0.25,
          'equation': 'ikc,jkc->ijc',
          'num_intermediate_channel': 128,
          'orientation': 'per_row',
          'shared_dropout': True
        },
        'triangle_multiplication_incoming': {
          'dropout_rate': 0.25,
          'equation': 'kjc,kic->ijc',
          'num_intermediate_channel': 128,
          'orientation': 'per_row',
          'shared_dropout': True
        },
        'pair_transition': {
          'dropout_rate': 0.0,
          'num_intermediate_factor': 4,
          'orientation': 'per_row',
          'shared_dropout': True
        }
      },
      'extra_msa_channel': 64,
      'extra_msa_stack_num_block': 4,
      'max_relative_feature': 32,
      'msa_channel': 256,
      'pair_channel': 128,
      'prev_pos': {
        'min_bin': 3.25,
        'max_bin': 20.75,
        'num_bins': 15
      },
      'recycle_features': True,
      'recycle_pos': True,
      'seq_channel': 384,
      'template': {
        'attention': {
          'gating': False,
          'key_dim': 64,
          'num_head': 4,
          'value_dim': 64
        },
        'dgram_features': {
          'min_bin': 3.25,
          'max_bin': 50.75,
          'num_bins': 39
        },
        'embed_torsion_angles': False,
        'enabled': False,
        'template_pair_stack': {
          'num_block': 2,
          'triangle_attention_starting_node': {
            'dropout_rate': 0.25,
            'gating': True,
            'key_dim': 64,
            'num_head': 4,
            'orientation': 'per_row',
            'shared_dropout': True,
            'value_dim': 64
          },
          'triangle_attention_ending_node': {
            'dropout_rate': 0.25,
            'gating': True,
            'key_dim': 64,
            'num_head': 4,
            'orientation': 'per_column',
            'shared_dropout': True,
            'value_dim': 64
          },
          'triangle_multiplication_outgoing': {
            'dropout_rate': 0.25,
            'equation': 'ikc,jkc->ijc',
            'num_intermediate_channel': 64,
            'orientation': 'per_row',
            'shared_dropout': True
          },
          'triangle_multiplication_incoming': {
            'dropout_rate': 0.25,
            'equation': 'kjc,kic->ijc',
            'num_intermediate_channel': 64,
            'orientation': 'per_row',
            'shared_dropout': True
          },
          'pair_transition': {
            'dropout_rate': 0.0,
            'num_intermediate_factor': 2,
            'orientation': 'per_row',
            'shared_dropout': True
          }
        },
        'max_templates': 4,
        'subbatch_size': 128,
        'use_template_unit_vector': False,
      }
    },
    'global_config': {
      'deterministic': False,
      'subbatch_size': 4,
      'use_remat': False,
      'zero_init': True
    },
    'heads': {
      'distogram': {
        'stop_gradient': False,
        'first_break': 2.3125,
        'last_break': 21.6875,
        'num_bins': 64,
        'weight': 0.3,
        'distillation': {
          'alpha': 0.0,
          'T' : 1.0,
         }
      },
      'predicted_aligned_error': {
        # `num_bins - 1` bins uniformly space the
        # [0, max_error_bin A] range.
        # The final bin covers [max_error_bin A, +infty]
        # 31A gives bins with 0.5A width.
        'stop_gradient': False,
        'max_error_bin': 31.,
        'num_bins': 64,
        'num_channels': 128,
        'filter_by_resolution': True,
        'min_resolution': 0.1,
        'max_resolution': 3.0,
        'weight': 0.0,
      },
      'experimentally_resolved': {
        'stop_gradient': False,
        'filter_by_resolution': True,
        'max_resolution': 3.0,
        'min_resolution': 0.1,
        'weight': 0.01,
        'distillation': {
          'alpha': 0.0,
          'T' : 1.0,
         }
      },
      'structure_module': {
        'stop_gradient': False,
        'num_layer': 8,
        'fape': {
          'clamp_distance': 10.0,
          'clamp_type': 'relu',
          'loss_unit_distance': 10.0
        },
        'angle_norm_weight': 0.01,
        'chi_dependent' : True, 
        'chi_weight': 0.5,
        'chi1_weight': 1.0,
        'chi2_weight': 1.0,
        'chi3_weight': 1.0,
        'chi4_weight': 1.0,
        'bb_weight': 0.5,
        'pair_weight': 0.0,
        'pair_epsilon': 0.0,
        'phipsi_lit_weight': 1.0,
        'phichi1_lit_weight': 1.0,
        'psichi1_lit_weight': 1.0,
        'chi1chi2_lit_weight': 0.0,
        'clash_overlap_tolerance': 1.5,
        'compute_in_graph_metrics': True,
        'dropout': 0.1,
        'num_channel': 384,
        'num_head': 12,
        'num_layer_in_transition': 3,
        'num_point_qk': 4,
        'num_point_v': 8,
        'num_scalar_qk': 16,
        'num_scalar_v': 16,
        'position_scale': 10.0,
        'secondary':{
            'num_output' : 8,
            'weight_frac' : 0.0,
            },
        'sidechain': {
          'atom_clamp_distance': 10.0,
          'num_channel': 128,
          'num_residual_block': 2,
          'weight_frac': 0.5,
          'length_scale': 10.,
          # jhw - weighting params for FAPE
          'weight_param_alpha': 1.5,
          'weight_param_h_offset': 10.0,
          'weight_param_v_offset': 0.5,
        },
        'structural_violation_loss_weight': 1.0,
        'violation_tolerance_factor': 12.0,
        'weight': 1.0
      },
      'predicted_lddt': {
        'stop_gradient': False,
        'filter_by_resolution': True,
        'max_resolution': 3.0,
        'min_resolution': 0.1,
        'num_bins': 50,
        'num_channels': 128,
        'weight': 0.01,
        'distillation': {
          'alpha': 0.0,
          'T' : 1.0,
         }
      },
      'predicted_sc_confidence': {
        'stop_gradient': False,
        'filter_by_resolution': True,
        'max_resolution': 3.0,
        'min_resolution': 0.1,
        'num_bins': 50,
        'num_channels': 128,
        'weight': 0.00,
      },
      'masked_msa': {
        'stop_gradient': False,
        'num_output': 23,
        'weight': 2.0,
        'distillation': {
          'alpha': 0.0,
          'T' : 1.0,
         }
      },
      'distillation': {
        'weight': 0.0,
        'input_pair': {
          'alpha': 0.0,
          'T': 1.0,
        },
        'msa': {
          'alpha': 0.0,
          'T': 1.0,
        },
        'msa_first_row': {
          'alpha': 0.0,
          'T': 1.0,
        },
        'pair': {
          'alpha': 0.0,
          'T': 1.0,
        },
        'ipa': {
          'alpha': 0.0,
          'T': 1.0,
        },
        'backbone': {
          'alpha': 0.0,
          'T': 1.0,
        },
        'single': {
          'alpha': 0.0,
          'T': 1.0,
        },
        'structure_module': {
          'alpha': 0.0,
          'T': 1.0,
        },
      }
    },
    'num_recycle': 10,
    'resample_msa_in_recycling': True
  },
})
