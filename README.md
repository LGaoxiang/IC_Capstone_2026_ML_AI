# IC_Capstone_2026_ML_AI

## Project Overview

This capstone project offers an in‑depth research into **Black‑Box Optimisation** (BBO), an important area of machine learning where the synthetic functions have no known internal fomular or logic.

* **Real-World ML Relevance**<br>In practical industrial and scientific contexts, evaluating such functions is "expensive" because it may require long runtimes, large amounts of computational resources and physical materials. The main objective is to interactively query eight distinct hidden functions (each modeling a different real-world scenario) and locate their global maximises within a limited number of iterations. BBO is essential in high-stakes settings such as detecting contamination sources, drug discovery (balancing compound combinations to reduce adverse effects), optimisation of placing products across warehouses, yield of a chemical process, optimising a cake recipe and deep learning hyperparameter tuning.
* **Career Support**<br>Solving BBO challenges equips me with practical techniques and decision-making skills under uncertainty, effectively connecting mathematical modelling with real-world engineering applications.

## Inputs and Outputs


### Inputs
* Inputs are supplied as high‑dimensional coordinate vectors in the form `x1-x2-x3-...-xn`.<br>
Example for a 8D function:<br>
0.387521-0.301058-0.396852-0.494557-0.550321-0.633894-0.487165-0.726984
* Every input coordinate must lie in the range **0.000000 to 0.999999** (i.e. greater than or equal to 0 and strictly less than 1) and have exactly six decimal places.<br>
* Function dimensionality ranges from 2 to 8.<br>
|Function|Dimensions|Initial Points|
|:----:|:----:|:---:|
|1|2D|10|
|2|2D|10|
|3|3D|15|
|4|4D|30|
|5|4D|20|
|6|5D|20|
|7|6D|30|
|8|8D|40|

### Outputs

* The system replies with a single scalar output value.
* All tasks are framed as maximisation problems (even if the real-world analogy is minimisation, as it's transformed so that higher is better).
* Outputs are used to refine the models and to further optimize the next inputs to achieve better output.<br>


## Challenge Objective


* The goal is to discover input configurations that **maximise** each of the eight black‑box functions. While the optimisation framework is designed for maximisation, the underlying applications vary: direct maximisation (e.g., F1 - locating probable radiation sources and F5) and minimisation via transformation (e.g., F3 - minimising the side effects of a new medicine). 
* Only **13 iterations** are allowed, so every query must be thoughtfully justified to extract maximum value. The unknown functions may be full of local optima (F4 - exhibiting several distinct peaks), may contain output noise (F2) and feature unknown internal dynamics – especially challenging in higher dimensions (F8 - up to 8D).

## Technical Approach


Balancing exploration-exploitation has been central to my strategy adjustments. Too much exploration wastes queries in low‑value regions, while too much exploitation may miss the global optimum.
* **Round 1: Baseline**<br>
Used a Gaussian Process + a fixed UCB (𝛽 = 1.96, fixed RBF length scale is 0.1 and noise assumption 1e-6)  for all functions.<br>
**Result:** Only worked for F2. Failed for F1, F3, F4, F5, F6, F7, F8.

* **Round 2: Unimprovement check + Grid search**<br>
Introduced an automatically switch to grid research (for low dimensions) and the alternation to EI to force exploration when the outputs does not improve for three queries.<br>
Reviewed and shifted to different length scales: 0.5 (F1, F3, F6) and 1.0 (F4, F5, F7).
Evaluated F8 across 20000 points randomly sampled in the dim*[0, 1] space due to the high dimension.<br>
**Result:** Worked for F3, F4, F5, F8. Failed for F1, F2, F6, F7

* **Round 3: Rand Transformation**<br>
Relied more on model predictions and still retained exploration mechanisms (increasing random candidates to 5000, 10000 and 20000 respectively on F6, F7 and F8).<br>
Some outputs of F1 were extremely small (e.g., 10⁻³¹²), causing numerical instability and leading the GP to recommend boundary points. To address this, I introduced a rank transformation to the preprocess of inputs.<br>
**Result:** Worked for F1, F2, F6, F7, F8. Failed for F3, F4, F5.

* **Round 4: WhiteKernel Noise Modeling & Trust Region Introduction**<br>
This round introduced two foundational improvements to the GP framework that would shape all subsequent queries:<br>
WhiteKernel for Noise Modeling: Replaced the fixed alpha noise assumption with a learnable WhiteKernel in the GP kernel for F1, F2, F3, F4, and F5. The standard configuration was WhiteKernel(`noise_level = 1e-5, noise_level_bounds = (1e-6, 1e-1)`) added to RBF. This allowed the GP to learn the appropriate noise level from data rather than assuming a fixed value.<br>
Trust Region Sampling: Introduced a hypercube trust region centered on the current best point. Initial radius = 0.1. Candidates were drawn 80% locally (inside the Trust Region) and 20% globally.<br>
**Result:** Worked for F1, F3, F4, F5, F8. Failed for F2, F6, F7.

* **Round 5: Candidate De-duplication & Trust Region Expansion Tuning**<br>
Building on Round 4's Trust Region framework, this round refined the sampling mechanics.<br>
Introduced `NearestNeighbors` filtering to exclude random candidates within 1e-6 of any already-evaluated point, preventing wasted queries on previously sampled regions.<br>
Lowered the consecutive-non-improvement threshold for TR expansion from 3 to 2, enabling faster exploration escape when stalled.<br>
For F5, switched active GP path from RBF + WhiteKernel to pure `Matern(nu=2.5)`.<br>
**Result:** Worked for F6, F7. Failed for F1, F2, F3, F4, F5, F8.

* **Round 6: Strategy Divergence**<br>
As queries accumulated, performance divergence across functions became pronounced. Began exploring neural network as alternative to GP on F6:
  - Architecture: Dense(64, relu) → Dense(32, relu) → Dense(1); SGD lr=0.1, epochs=300;
  - Gradient ascent: 50 steps, step 0.01, 5 restarts), UCB = mean + β × 0.1.
  - Trust Region sampling.<br>

  And auto-triggered for F4 after 3 consecutive unimproved queries.<br>
**Result:** Worked for F4, F6, F7. Failed for F1, F2, F3, F5, F8.

#### Neural Network Hyperparameters
In this Capstone project, the following hyperparameters were critical:
- Learning rate
- Hidden layer architecture
- Constant std for UCB<br>

**Continuous Hyperparameters** include learning rate and constant std for UCB. These are tuned via Bayesian optimisation or random search.<br> Number of hidden layers, number of neurons per layer, activation function and optimiser type are **Discrete yperparameters**. These are often tuned via grid search or manual trial.

* **Round 7: Matern Kernel Adoption**<br>
  - F1: Switch to Matern Kernel: Replaced RBF with Matern(nu=1.5).<br>
  - F2: Continued GP + RBF + WhiteKernel.<br>
  - F3: GP + RBF + WhiteKernel + Trust Region + normalize_y.<br>
  - F5: Continued Matern(nu=2.5) + normalize_y.<br>
  - F6: Continued NN surrogate.<br>
  - F7, F8: Continued GP.<br>

  **Result:** Worked for F1, F5, F8. Failed for F2, F3, F4, F6, F7.

* **Round 8: Log Transform**<br>
Major preprocessing upgrade for F1: Log Transform replaced Rank Transform, and the kernel was upgraded to ConstantKernel × Matern + WhiteKernel.<br>
F4 auto-triggered grid search and EI acquisition.<br>
**Result:** Worked for F1, F5, F6, F8. Failed for F2, F3, F4, F7.

* **Round 9: NN Surrogate for F7 and F8**<br>
  - F1: Log transform + Constant × Matern GP.<br>
  - F2: Switched to EI + Forced Grid Search.<br>
  - F7, F8: Officially Switched to NN Surrogate.<br>

  **Result:** Worked for F4, F5, F8. Failed for F1, F2, F3, F6, F7.

* **Round 10: Hyperparameter Tightening**<br>
With strategies now well-differentiated, this round focused on targeted hyperparameter tuning. Tightened F1 Matern length_scale bounds from (0.05, 50) to (0.01, 10) to resolve `ConvergenceWarning`. And WhiteKernel noise_level lower bound loosened to 1e-4.<br>
**Result:** Worked for F5, F8. Failed for F1, F2, F3, F4, F6, F7. No significant improvement.

* **Round 11: Grid Search Revival for Low-Dimensional Functions**<br>
  - F1: Manual Grid Search Activation: Exhaustive 100×100 grid with maximizing UCB on GP posterior. Grid search GP kernel upgraded to ConstantKernel × Matern + WhiteKernel.<br>
  - F2: Continued EI + grid search on 100×100 mesh.<br>
  - F4: Grid branch upgraded from RBF to Matern + WhiteKernel. Switched back to UCB acquisition (from E)<br>
  - F5: Matern GP continued.<br>
  - F6, F7, F8: NN surrogate continued with gradient ascent.<br>

  **Result:** Worked for F1, F5. Failed for F2, F3, F4, F6, F8..

* **Round 12: Final Convergence**<br>
In the final query round, targeted searches were directed based on each function's accumulated characteristics.<br>
**Result:** Only worked for F5. Failed for F1, F2, F3, F4, F6, F7, F8.
