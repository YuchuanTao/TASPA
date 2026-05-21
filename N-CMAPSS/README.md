# TASPA: TASPA: A network incorporating temporal and spatial prior information for predicting the remaining useful life of aircraft engines

This repository provides the implementation of **TASPA**, a
neural network that incorporates **temporal and spatial prior
information** into an attention-based architecture for predicting the
**Remaining Useful Life (RUL)** of aircraft engines.

Different from conventional purely data-driven approaches that rely on
increasingly complex network structures, TASPA explicitly integrates
prior knowledge embedded in the degradation process, including
degradation causality, operating conditions, and sensor coupling
relationships. These priors are used to modulate the attention
mechanism, leading to improved prediction accuracy and robustness.

------------------------------------------------------------------------

## 1. Repository Structure

    .
    ├── trained_model/
    ├── data_process.py
    ├── model.py
    ├── train.py
    ├── test.py
    └── README.md


------------------------------------------------------------------------

## 2. Requirements

-   Python  3.13.1
-   PyTorch  2.7.1+cu128
-   NumPy 2.2.0
-   Matplotlib 3.10.0

``` bash
pip install torch numpy matplotlib
```

------------------------------------------------------------------------

## 3. Dataset Description

Since the N-CMAPSS dataset is too large to upload, please download it yourself.


------------------------------------------------------------------------

## 4. Data Preprocessing

Implemented in `data_process.py`, including:

-   Informative sensor selection\
-   Z-score normalization\
-   Sliding window sample construction\
-   Capped RUL label generation

------------------------------------------------------------------------

## 5. Model Overview

### Temporal Prior Information

-   Causal masking
-   Operating-condition-aware attention modulation

### Spatial Prior Information

-   Sensor coupling modeled as a distance matrix
-   Distance-aware Transformer attention

------------------------------------------------------------------------

## 6. Training

``` bash
python train.py
```

Trained models are saved to:

    ./trained_model/

------------------------------------------------------------------------

## 7. Testing and Visualization

``` bash
python test.py
```

The script plots sorted true RUL and predicted RUL for comparison.

------------------------------------------------------------------------

## 8. Reproducibility

-   Fixed random seeds\
-   Relative paths\
-   Consistent preprocessing

------------------------------------------------------------------------

## 9. Citation

TASPA: A Network Incorporating Temporal and Spatial Prior Information
for Predicting the Remaining Useful Life of Aircraft Engines.

------------------------------------------------------------------------

## 10. License

For academic and research use only.

------------------------------------------------------------------------

## 11. Contact

**First Author**\
Yuchuan Tao\
Email: 1150324744@qq.com
or tao1150324744@gmail.com
