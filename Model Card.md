# Overview

**Name:** Adaptive Bayesian Optimisation with Trust Regions and Neural Surrogates <br>
**Type:** Black‑Box Optimisation (BBO) framework <br>
**Version:** 1.0

# Intended Use

**Suitable tasks:** <br> Maximisation of continuous‑domain functions with a limited evaluation budget. Particularly effective for low‑to‑moderate dimensions. <br>
**Use cases to avoid:**
* Optimisation of discrete or combinatorial spaces.
* Problems where the function is highly discontinuous.

# Details

Over ten rounds, the strategy evolved through the following stages: <br>
* **Round 1–2:** Fixed RBF kernel, constant β=1.96, random candidate sampling.
* **Round 3–4:** Introduced rank transformation for extreme outputs of F1 and unimprovement detection. Switched to grid search for low‑dimension functions.
* **Round 5–6:** Replaced RBF with Matern kernel for sharp peaks (F5), added trust‑region sampling and historical‑point exclusion to eliminate duplicates and boundary clustering.
* **Round 7–8:** Extended trust‑region logic with adaptive radius and separate handling of high‑dimensional functions (F6) using neural network surrogates with gradient‑ascent acquisition optimisation.
* **Round 9–10:** Fine‑tuned hyperparameters (candidate counts, global ratios, restart numbers) and finalised the layered strategy: grid search (<4D), trust‑region + GP (4D), trust‑region + NN (5–8D).

# Performance

* **Low‑dim (F1–F3):** Grid search yielded the best maxima; F1’s extreme range was successfully navigated with rank/standardisation.

* **Mid‑dim (F4–F5):** Trust‑region + Matern GP consistently improved best values, capturing F5’s sharp yield peak.

* **High‑dim (F6–F8):** Neural network surrogates provided viable recommendations.

# Assumptions and Limitations

**Key assumptions:**
* Functions are locally smooth.
* The global optimum lies within the explored region or can be reached by gradual trust‑region expansion.

**Limitations:**
* Trust‑region radius and shrink/expand factors are manually set, lacking global adaptivity.
* Neural network surrogates require careful seed control and deterministic operations for reproducibility.
* In extremely high dimensions or with needle‑thin peaks, the method may miss the global optimum due to sparse sampling and smoothness priors.

# Ethical Considerations

Transparency is ensured through open‑source code, detailed documentation and persistent logging of all queries and results. This supports full reproducibility. The explicit recording of assumptions and failure modes fosters responsible use, preventing over‑claiming of optimality.
