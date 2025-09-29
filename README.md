# IO-LVM
This project is related to the NeurIPS 2025 paper "Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems". 


## Main files for training and saving models:

### `main_spp_synthetic.py`
Related to the Waxman generated graphs and paths experiment. Here you can set the input "mult" to 1 or to 0 to choose between the multiple start/target nodes versus the single start/target nodes experiment.

### `main_spp_cabspot.py`
Related to the taxi trajectories experiment. Please extract the data found in "cabspotting_preprocessing" folder.

### `main_spp_ship.py`
Related to the ship trajectories experiment.

### `main_tsp_tsplib.py`
Related to the tsp_lib experiment with Hamiltonian cycles generated data.

**Main Arguments:**
1. `latent_dim`: Number of latent dimensions, check the paper for a better grasp on values to choose.
2. `method`: Most important values are "IOLVM" and "VAE".
3. `alpha_kl` or `beta`: KL regularization. Depending on the experiment you might correct the value according to the batch size (e.g., whatever is given in the paper/BS).
4. `n_epochs`: Number of epochs.
5. `eps`: The perturbation for gradient estimation, 0.05 generally works fine.
6. `lr`: Learning Rate, defaults work fine.


## Jupyter file to explore saved models (synthetic example)

### `compare_synthetic_eval.ipynb`

Related to the Waxman generated graphs and paths experiment. I saved a IOLVM .pkl and a VAE .pkl for a comparison purpose, but feel free to train with different parameters and try it yourself.

---
