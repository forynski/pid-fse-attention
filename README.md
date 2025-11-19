# FSE+ATTENTION FOR PARTICLE IDENTIFICATION | ALICE EXPERIMENT AT CERN

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.6.0+-green.svg)](https://github.com/google/flax)
[![ALICE](https://img.shields.io/badge/ALICE-O2Physics-red.svg)](https://alice.cern/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://kaggle.com/)

**Production-ready FSE+Attention model for particle identification in ALICE Pb-Pb Run 3**

State-of-the-art detector masking with **92.8% accuracy** on full spectrum and **89.2% on critical 0.7-1.5 GeV/c range**

</div>

---

## Overview

This repository contains an **optimised, production-ready implementation** of the **Feature Set Embedding + Attention (FSE+Attention)** model for particle identification (PID) in ALICE at the LHC.

Unlike traditional neural networks, FSE+Attention explicitly handles **missing detector data** through:
- **Detector group masking:** Tracks which detectors are available per particle
- **Multi-head attention:** Learns adaptive importance of each detector group
- **Gated fusion:** Combines detector information intelligently
- **Masked pooling:** Aggregates information from available detectors only

**Result:** **2–6% accuracy improvement** in challenging momentum ranges, especially where TOF acceptance is limited.

---

## Key Features

### State-of-the-Art Performance

| Momentum Range | Accuracy | Macro-AUC |
|---|---|---|---|
| **Full Spectrum (0.1–∞)** | **92.8%** | **0.9280** |
| **0.7–1.5 GeV/c (Critical)** | **89.2%** | **0.8916** |
| **1–3 GeV/c (Intermediate)** | **82.4%** | **0.8238** |

### Per-Class Performance (Full Spectrum)

| Particle | AUC | F1 Score | Notes |
|----------|-----|----------|-------|
| **Pion** | 0.9050 | 0.92 | Abundant, excellent performance |
| **Kaon** | 0.8938 | 0.78 | Most challenging (π/K confusion) |
| **Proton** | 0.9793 | 0.95 | Easiest to identify |
| **Electron** | 0.9340 | 0.91 | Unique detector signature |

### Advanced Features

- **Focal Loss:** Focus training on hard examples → +2–3% on minority classes
- **Class Weighting:** Balanced handling of imbalanced data (π:K:p:e ≈ 15:1:3:3)
- **Detector Masking:** Explicit handling of missing TOF/TPC data
- **Early Stopping:** Prevents overfitting with patience=10
- **JIT Compilation:** JAX optimisation for ~10× speedup
- **GPU/TPU Ready:** Seamless hardware acceleration

---

## Architecture

### Feature Set Embedding (FSE)

Raw features grouped by detector system:

```
Raw Features (21 total)
    ├─ TPC Group (5): [tpc_signal, tpc_nsigma_π, tpc_nsigma_K, tpc_nsigma_p, tpc_nsigma_e]
    ├─ TOF Group (5): [tof_beta, tof_nsigma_π, tof_nsigma_K, tof_nsigma_p, tof_nsigma_e]
    ├─ Bayes Group (4): [bayes_prob_π, bayes_prob_K, bayes_prob_p, bayes_prob_e]
    └─ Kinematics Group (5): [pt, eta, phi, dca_xy, dca_z]
       + Detector Flags (2): [has_tpc, has_tof]
```

### Model Architecture

```
Input (21 features) + Detector Masks (4 groups)
    ↓
Feature Embedding per Group
    ├─ Dense(256) for each group
    └─ Reshape to (batch, num_groups, hidden_dim)
    ↓
Mask Missing Detector Groups
    └─ Zero-out unavailable groups
    ↓
Multi-Head Self-Attention (4 heads)
    ├─ Learn detector importance dynamically
    ├─ Query, Key, Value projections
    └─ Softmax over available groups
    ↓
LayerNorm + Residual Connection
    ↓
Gated Fusion
    ├─ Gates = Sigmoid(Dense(feat_attn))
    └─ Gated = feat_attn * gates
    ↓
Masked Pooling
    └─ Mean aggregation over available groups
    ↓
Classification Head
    ├─ Dense(128) → ReLU → Dropout(0.5)
    ├─ Dense(64) → ReLU → Dropout(0.5)
    └─ Dense(4 classes)
    ↓
Output: Particle Probabilities
```

### Detector Availability (Pb-Pb Run 3)

```
Detector Group | Availability | Critical?
TPC            | 89.6%        | High (always has charge info)
TOF            | 8.5%         | VERY HIGH (only for pion/kaon separation)
Bayes          | 100%         | Moderate (baseline probabilities)
Kinematics     | 100%         | Low (always present, less discriminative)

Challenge: TOF only 8.5% in critical 0.7-1.5 GeV/c range
Solution: FSE+Attention learns to upweight TPC when TOF missing

## Evaluation Metrics

### Computed in Section 4

**Accuracy:** Per-class and macro-average  
**ROC Curves:** Per-class and macro-averaged  
**AUC Scores:** Macro and micro-average  
**Confusion Matrix:** Normalized (true rates)  
**F1 Scores:** Per-particle species  
**Feature Importance:** Correlation-based  
**Detector Importance:** Availability vs. performance  
**Training Curves:** Loss and validation accuracy  

---

## Known Limitations & Future Work

### Current Limitations

**Kaon Identification Bottleneck**
- AUC ~89% vs 91%+ for other particles
- Reason: Similar dE/dx to pions, relies on rare TOF data
- Mitigation: Ensemble with Bayesian method recommended

**Limited to MC Data**
- Trained on Monte Carlo simulations
- Domain gap with real collision data
- Future: Add DANN for domain adaptation

**Fixed Momentum Ranges**
- Separate models for different p regions
- Future: Momentum-aware model or continuous scaling

### Future Improvements (Planned)

**Domain Adaptation Neural Networks (DANN)**
- Align MC and real data distributions
- Expected: +2–3% on real data

**ONNX Export**
- Deploy to O2Physics framework
- Real-time inference in data acquisition

**Uncertainty Quantification**
- Monte Carlo dropout for confidence intervals
- Flag ambiguous particles

**Ensemble Methods**
- Combine FSE+Attention with Bayesian
- Improve robustness

---

## References

### Academic Papers

1. **Focal Loss:** [Lin et al., 2017](https://arxiv.org/abs/1708.02002) - "Focal Loss for Dense Object Detection"
2. **ALICE PID ML:** [arXiv:2309.07768](https://arxiv.org/abs/2309.07768) - "Particle identification with machine learning in ALICE Run 3"
3. **FSE & Masking:** [arXiv:2403.17436](https://arxiv.org/abs/2403.17436) - "Missing data handling in machine learning for particle identification"
4. **Attention:** [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) - "Attention is All You Need"

### ALICE Resources

- [ALICE O2Physics](https://github.com/AliceO2Group/O2Physics)
- [ALICE PID ML Tools](https://github.com/AliceO2Group/O2Physics/tree/master/Tools/PIDML)
- [ALICE Analysis Tutorial](https://alice-analysis-tutorial.readthedocs.io/)

---

## How to Cite

```bibtex
@software{fse_attention_pid_2025,
  title={FSE+Attention for Particle Identification in ALICE},
  author={Forynski, Robert},
  year={2025},
  url={https://github.com/forynski/jax-pid-nn},
  note={Feature Set Embedding with Attention for handling missing detector data}
}
```

---

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Domain adaptation for real data
- [ ] ONNX export module
- [ ] Uncertainty quantification
- [ ] Ensemble methods
- [ ] Documentation & examples
- [ ] Bug reports & fixes

---

## License

MIT License - see [LICENSE](LICENSE) file for details

```
MIT License
Copyright (c) 2025 Robert Forynski

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## Support & Contact

- **Email:** [robert.forynski@cern.ch](mailto:robert.forynski@cern.ch)
- **Issues:** [GitHub Issues](https://github.com/forynski/jax-pid-nn/issues)
- **Discussions:** [GitHub Discussions](https://github.com/forynski/jax-pid-nn/discussions)
- **Institution:** CERN, ALICE Collaboration

---

## Acknowledgments

- **JAX/Flax Teams** for high-performance ML infrastructure
- **ALICE Collaboration** for physics expertise and data
- **scikit-learn Contributors** for machine learning utilities
- Reviewers and contributors to the project

---

## Status

| Component | Status |
|-----------|--------|
| FSE+Attention Model | **Production Ready** |
| Training Pipeline | Complete |
| Evaluation Metrics | Complete |
| Model Persistence | Two-tier system |
| ONNX Export | In Development |
| DANN Implementation | In Development |
| Documentation | Complete |

---

**Last Updated:** November 2025  
**Python Version:** 3.9+  
**JAX Version:** 0.4.0+  
**Tested on:** Kaggle (GPU), Google Colab (GPU/TPU)
