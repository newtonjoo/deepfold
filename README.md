# Protein 3D Structure Prediction with DeepFold

> We have developed a new pipeline of protein structure prediction called **DeepFold**, that improves the accuracy of side-chain predictions as well as that of backbones by leveraging AlphaFold2.

- First, we optimized the loss functions of side chains by considering the sequential dependence of each torsion angle.
- Second, we enhanced template features to capture better structural context between residue pairs by employing advanced sequence alignment methods and exploring the structure database. 
- Last, we implemented a reoptimization step that utilizes the energy function of molecular mechanics and an advanced global optimization method to enhance the structural validity of the prediction.

This package provides an implementation of **DeepFold**, a trainable, Transformer-based deep protein folding model. We modified the open-source code of [DeepMind AlphaFold v2.0](https://github.com/deepmind/alphafold) and [Uni-Fold-jax](https://github.com/dptech-corp/Uni-Fold-jax). 


## 1. Install the environment.
We recommend using [Docker](https://www.docker.com/) to install the environment. The Dockerfile is provided in the `docker` folder. To build the docker image, run the following command:

```bash
  git clone https://github.com/newtonjoo/deepfold
  cd deepfold

  # build docker
  docker build -f docker/Dockerfile -t deepfold .
```

Mount the data folder on the docker and attach to bash.
```bash
  docker run --gpus all -v "${DATA_DIR}:/app/deepfold/data" -it deepfold:latest
```

## 2. Inference with trained models.
We provide the trained models on the [google drive](https://drive.google.com/drive/folders/1Qyq0oZo4NUv55X7N7LSjqJOZcmR23NV6?usp=sharing). Please download the models and put them in the `data/params` folder.


### 2.1 Infer from features.pkl.

We provide the [`run_from_pkl.py`](./run_from_pkl.py) script to support inferring protein structures from `features.pkl` inputs. A demo command would be

```bash
python run_from_pkl.py \
  --pickle_paths ./example_data/features/aa/1aac_1_A/features.pkl \
  --model_names model1 \
  --model_paths params/model1.npz \
  --output_dir ./out
```

The command will generate structures (in PDB format) from input features predicted by different input models, the running time of each component, and corresponding residue-wise confidence score (predicted LDDT, or pLDDT).

### 2.2 Infer from FASTA files.

Essentially, inferring the structures from given FASTA files includes two steps, i.e. generating the pickled features and predicting structures from them. We provided a script, [`run_from_fasta.py`](./run_from_fasta.py), as a friendlier user interface. An example usage would be

```bash
python run_from_pkl.py \
  --fasta_paths ./example_data/fasta/1aac_1_A.fasta \
  --model_names model1 \
  --model_paths params/model1.npz \
  --data_dir /path/to/database/directory
  --output_dir ./out
```

## 3. Manual training

If you want to train the model from scratch, you can use the docker image to run the training script. The training script is `run_train.sh`. You can modify the script to suit your learning settings.

### 3.1 Prepare data before training.

Before you start to train your own folding models, you shall prepare the features and labels of the training proteins. Features of proteins mainly include the amino acid sequence, MSAs and templates of proteins. These messages should be contained in a pickle file `<name>/features.pkl` for each training protein. Deepfold provides scripts to process input FASTA files, relying on several external databases and tools. Labels are CIF files containing the structures of the proteins.

### 3.2 Datasets and external tools.

Deepfold adopts the same data processing pipeline as AlphaFold2. We kept the scripts of downloading corresponding databases for searching sequence homologies and templates in the AlphaFold2 repo. Use the command

```bash
  bash scripts/download_all_data.sh /path/to/database/directory
```

to download all required databases of Deepfold.

If you successfully installed the Conda environment in Section 1, external tools of search sequence homologies and templates should be installed properly. As an alternative, you can customize the arguments of the feature preparation script (`generate_pkl_features.py`) to refer to your own databases and tools.

### 3.3 Run the preparation code.

An example command of running the feature preparation pipeline would be

```bash
  python generate_pkl_features.py \
    --fasta_dir ./example_data/fasta \
    --output_dir ./out \
    --data_dir /path/to/database/directory \
    --num_workers 1
```

This command automatically processes all FASTA files under `fasta_dir`, and dumps the results to `output_dir`. Note that each FASTA file should contain only one sequence. The default number of CPUs used in hhblits and jackhmmer are 4 and 8. You can modify them in `deepfold/data/tools/hhblits.py` and `deepfold/data/tools/jackhmmer.py`, respectively.

### 3.4 Organize your training data.

Deepfold uses the class [`DataSystem`](./deepfold/train/data_system.py) to automatically sample and load the training proteins. To make everything goes right, you shall pay attention to how the training data is organized. Two directories should be established, one with input features (`features.pkl` files, referred to as `features_dir`) and the other with labels (`*.cif` files, referred to as `mmcif_dir`). The feature directory should have its files named as `<pdb_id>_<model_id>_<chain_id>/features.pkl`, e.g. `101m_1_A/features.pkl`, and the label directory should have its files named as `<pdb_id>.cif`, e.g. `101m.cif`. See [`./example_data/features`](./example_data/features) and [`./example_data/mmcif`](./example_data/mmcif) for instances of the two directories. Notably, users shall make sure that all proteins used for training have their corresponding labels. This is checked by `DataSystem.check_completeness()`.


### 3.5 Configuration.
Before you conduct any actual training processes, please make sure that you correctly configured the code. Modify the training configurations in [`deepfold/train/train_config.py`](./deepfold/train/train_config.py). We annotated the default configurations to reproduce AlphaFold in the script. Specifically, modify the configurations of data paths:
    
  ```json
  "data": {
    "train": {
      "features_dir": "where/training/protein/features/are/stored/",
      "mmcif_dir": "where/training/mmcif/files/are/stored/",
      "sample_weights": "which/specifies/proteins/for/training.json"
    },
    "eval": {
      "features_dir": "where/validation/protein/features/are/stored/",
      "mmcif_dir": "where/validation/mmcif/files/are/stored/",
      "sample_weights": "which/specifies/proteins/for/training.json"
    }
  }
  ```
  
  The specified data should be contained in two folders, namely a `features_dir` and a `mmcif_dir`. Organizations of the two directories are introduced in Section 2.3. Meanwhile, if you want to specify a subset of training data under the directories, or assign customized sample weights for each protein, write a json file and feed its path to `sample_weights`. This is optional, as you can leave it as `None` (and the program will attempt to use all entries under `features_dir` with uniform weights). The json file should be a dictionary containing the basenames of directories of protein features ([pdb_id]\_[model_id]\_[chain_id]) and the sample weight of each protein in the training process (integer or float), such as:

  ```json
  {"1am9_1_C": 82, "1amp_1_A": 291, "1aoj_1_A": 60, "1aoz_1_A": 552}
  ```
  or for uniform sampling, simply using a list of protein entries suffices:

  ```json
  ["1am9_1_C", "1amp_1_A", "1aoj_1_A", "1aoz_1_A"]
  ```

For users who want to customize their own folding models, configurations of model hyperparameters can be edited in [`deepfold/model/config.py`](./deepfold/model/config.py) .

### 3.6 Run the training code!
To train the model on a single node without MPI, run
```bash
python train.py
```

You can also train the model with multiple GPUs using MPI (or workload managers that supports MPI, such as PBS or Slurm) by running:
```bash
sbatch run_train.sh
```

In either way, make sure you properly configurate the option `use_mpi` and `gpus_per_node` in [`deepfold/train/train_config.py`](./deepfold/train/train_config.py).

## 4. License and disclaimer.

### 4.1 DeepFold code license.

Copyright 2022 DeepFold Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at 
<http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

### 4.2 Use of third-party software.

Use of the third-party software, libraries or code may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.
