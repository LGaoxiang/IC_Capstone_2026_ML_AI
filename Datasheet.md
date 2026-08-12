# Motivation

This dataset was created to support the Black‑Box Optimisation (BBO) capstone project, which aims to find the global maximum of eight unknown synthetic functions. Each function represents a different real‑world optimisation challenge. The dataset captures all queries submitted to the hidden functions and their corresponding outputs, enabling iterative surrogate modelling and acquisition‑function‑based search. The task is maximisation of a noisy black‑box function with a limited evaluation budget of 13 queries per function.

# Composition

The dataset contains, for each of the eight functions (F1–F8):

* Inputs: a matrix of shape `(N, D)` where `N` is the total number of queries submitted so far (initial + user submissions) and `D` is the dimensionality. All input values lie in [0, 0.999999] with six‑decimal precision.
* Outputs: a vector of length N containing the scalar response value. Outputs can be positive, negative, or extremely small (e.g., 1e‑312).
* Auxiliary data: trust‑region states.<br>

Files are stored in NumPy .npy format.

# Collection Process

* **Initial data** was provided by the capstone system.
* **Subsequent queries** were chosen by maximising an acquisition function (UCB or EI) on a Gaussian Process (GP) or Neural Network surrogate.
* The strategy evolved from a fixed‑kernel GP + random candidates to an adaptive system with trust regions, grid search for low dimensions, historical‑point exclusion, Matern kernels and Neural Network.
* The collection timeframe spanned thirteen rounds over several weeks, with one submission batch per round.

# Preprocessing and Uses

**Preprocessing applied:**
* Output standardisation (`normalize_y=True` in GP) or rank transformation for extreme numeric ranges (F1).
* Input clamping to `[0, 0.999999]` and exclusion of previously queried points.
* Fixed noise level or kernel switching to Matern to resolve convergence warnings.<br>

**Intended uses:** surrogate‑model training, acquisition‑function optimisation, performance monitoring and strategy evaluation.

# Distribution and Maintenance

The dataset is available in the project’s GitHub repository under the `data/` and `results/` directories. The dataset is maintained by the project author and updated after each submission round.
