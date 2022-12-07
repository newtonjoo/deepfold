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

"""Configurations for training DeepFold."""

import copy
from ml_collections import ConfigDict

def custom_train_config(name: str) -> ConfigDict:
  """Get the ConfigDict of a train setting."""
  if name not in CONFIG_DIFFS:
    raise ValueError(f'Invalid config name {name}.')
  cfg = copy.deepcopy(train_config)
  cfg.update_from_flattened_dict(CONFIG_DIFFS[name])
  return cfg

CONFIG_DIFFS = {
  # Users may create customized configs by adding entries in this dict
  # and referring to them via the key name.
  'example': {
      'optimizer.learning_rate': 1e-3,
      'optimizer.warm_up_steps': 1000,
      'optimizer.decay.name': 'exp',
  },
}


train_config = ConfigDict({
    'global_config':{
        # build distillation data set
        'build_distillation': False,
        # training using distillation data set
        'train_distillation': True,
        # gradient batch per gpu
        'minibatch_size': 1, 
        # gradient accumulation
        'gradient_accumulation': 1, 
        # dataloader num worker
        'data_num_workers': 16, 
        # Max queue size. Specifies the queue size of the pre-processed
        # batches. Generally has little impact on code efficiency.
        'max_queue_size': 16,
        # whether you are using MPI communication for multi-gpu training.
        'use_mpi': True,
        # This specifies a model config defined in `deepfold/model/config.py`. 
        # You can also customize your own model config 
        # in `deepfold/model/config.py` and specify it here.
        'model_name': 'model1',
        # Verbosity of logging messages.
        'verbose': 'info',
        # The number of processes/gpus per node
        'gpus_per_node': 8,
        # The format for autoloading the checkpoint, choose from 'pkl' and 
        # 'npz'. Note that `pkl` format stores necessary variables of 
        # optimizers, yet `npz` saves only model parameters.
        'ckpt_format': 'npz',
        # Initial step. if > 0, the model will auto-load ckpts from `load_dir`.
        'start_step': 1,                # 0 by default
        # Max steps for training. Accumulated from 'start_step' instead of 0.
        'end_step': 15000,                # 80000 in af2
        # Frequency of logging messages and the training loss curve.
        'logging_freq': 100, #1000
        # Frequency of validation.
        'eval_freq': 10, #500
        # Frequency of saving ckpts.
        'save_freq': 1000,
        # Directory to save ckpts. used for auto-saving ckpts.
        'save_dir': './out/ckpt',
        # Directory to load ckpts. used for auto-loading ckpts.
        # ignored if start_step == 0.
        'load_dir': './out/ckpt',
        # Training precision, generally in ['fp32', 'bf16'].
        # Set for mixed precision training.
        'precision': 'fp32',
        # Random seed for initializing model parameters. Ignored when attempting to auto load ckpts.
        'random_seed': 181129
    },
    'optimizer': {
        # Optimizer class.
        'name': 'adam',                 # in ['adam', 'sgd', ...]
        # Learning rate. if warm up steps > 0, this specifies the peak learning rate. 
        'learning_rate': 1e-3,          # 1e-3 in af2
        # The number of warm-up steps.
        'warm_up_steps': 0,            # 1000 in af2
        # Learning rate decay configs.
        'decay':{
            'name': 'piecewise_constant',              # in ['exp', 'piecewise_constant'..]
            #'decay_rate': 0.95,         # 0.95 in af2
            #'decay_steps': 10000           # 5000? in af2
            'boundaries': [5000,10000],
            'decays': [1e-3,1e-4,1e-5],
        },
        # Global clip norm of gradients.
        'clip_norm': 1e-1,
    },
    'data':{
        'train': {
            # Directory to store features (features.pkl files)
            'features_dir': "./example_data/features",
            # Directory to store labels (.mmcif files)
            'mmcif_dir': "./example_data/mmcif",
            # Directory to store secondary_structure (.pkl file)
            'dssp_dir' : "./example_data/dssp",
            # Json file that specifies sampling weights of each sample.
            'sample_weights': "./examle_data/train_weights.json"
        },
        'eval': {
            # Directory to store features (features.pkl files)
            'features_dir': "./example_data/features",
            # Directory to store labels (.mmcif files)
            'mmcif_dir': "./example_data/mmcif",
            # Directory to store secondary_structure (.pkl file)
            'dssp_dir' : "./example_data/dssp",
            # Json file that specifies sampling weights of each sample.
            'sample_weights': "./examle_data/train_weights.json"
        },
    },
    'distillation': {
        # build configs
        'build': {
            # Directory to store features (features.pkl files)
            'features_dir': "./example_data/features",
            # Directory to store labels (.mmcif files)
            'mmcif_dir': "./example_data/mmcif",
            # Json file that specifies sampling weights of each sample.
            'sample_weights': "./examle_data/train_weights.json",
            # Directory to store logits and values
            'save_dir': "./example_data/256/outputs", 
            # save pkl infos in json
            'build_list': './example_data/build_list.json',
        },
        # train config:
        'train':{
            # learn: train data list
            'train_data_list': './example_data/256/train_data_list.json',
            # learn: eval data list
            'eval_data_list': './example_data/256/eval_data_list.json',
            # Directory to store secondary_structure (.pkl file)
            'dssp_dir' : "./example_data/dssp_0328",
        }
    }
})
