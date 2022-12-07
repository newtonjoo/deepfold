# inference test 

import unittest
from unittest import TestCase
import logging
import pickle
import glob

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
import jax.numpy as jnp
import jax.random as jrand

from deepfold.model.config import model_config
from deepfold.train.utils import load_params
from deepfold.model.model import RunModel
from deepfold.common.protein import from_prediction, to_pdb
rng = jax.random.PRNGKey(42)

from tests.utils import get_tm

param_paths = [f"params/model{i}.npz" for i in range(1, 6)]
pkl_path = glob.glob("example_data/**/features.pkl", recursive=True)[0]
native_pdb_path = glob.glob("example_data/**/*.pdb", recursive=True)[0]

class TestInference(TestCase):
  def __init__(self, *args):
    super().__init__(*args)
  
  def set_model(self, model_num):
    self.config = model_config(f"model{model_num+1}")
    self.model_params = load_params(param_paths[model_num])
    self.model_runner = RunModel(config=self.config, params=self.model_params, num_recycle=3, ensemble_representations=False)

  def run_inference(self):
    with open(pkl_path, 'rb') as fp:
      features = pickle.load(fp)
    processed_features = self.model_runner.process_features(features, random_seed=0)
    prediction_result = self.model_runner.predict(processed_features)
    unrelaxed_pdb_str = to_pdb(from_prediction(processed_features, prediction_result))
    with open(f"tests/inference_temp.pdb", 'w') as fp:
      fp.write(unrelaxed_pdb_str)
    tm = get_tm(f'tests/inference_temp.pdb', native_pdb_path)
    os.remove(f"tests/inference_temp.pdb")

    log.info(f"InferenceTest: (tm={tm})")
    self.assertGreater(tm, 0.70)
  
  def test_models(self):
    for i in [0, 1, 2, 3, 4]:
      self.set_model(i)
      self.run_inference()
