# Repository for NeurIPS2021 submission: "Dual Decomposition of Convex Optimization Layer for Consistent Attention in Medical Images"

### 1. Download BraTS18 dataset
Apply for a data request at: https://www.med.upenn.edu/sbia/brats2018/registration.html.
### 2. Configuration
Edit `ROOT_DIR`, `DATA_DIR` and `PROJCET_DIR` in `config.py` for your working directories.
### 3. Processing the BraTS18 dataset
Run `dataio/create_dataset.py`.
### 4. Training
Run `train.py`.
### 5. Evaluation
Run `evals/segment.py`.