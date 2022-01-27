# Repository for ICML2022 submission: "Dual Decomposition of Convex Optimization Layer for Consistent Attention in Medical Images"

### 1. Download the BraTS18 dataset
Apply for a data request at: https://www.med.upenn.edu/sbia/brats2018/registration.html.
### 2. Setup
Using python 3.8, set up an environment using the `requirements.txt` file.
### 3. Project configuration
Edit `ROOT_DIR`, `DATA_DIR` and `PROJCET_DIR` in `config.py` for your working directories.
### 4. Processing the BraTS18 dataset
Run `dataio/create_dataset.py`.
### 5. Training
Run `train.py`.
### 6. Evaluation
Run `evals/segment.py`.
