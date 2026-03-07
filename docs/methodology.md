# Methodology — MMM Multi-Market Bayesian

## 1. Model Equation

```
Revenue(t) = BaseSales
           + Σ[β_i × Adstock(Spend_i(t)) × Hill(Spend_i(t))]
           + Seasonality(t)
           + Trend(t)
           + Events(t)
           + ε(t)
```

## 2. Adstock Transformation (Carryover Effect)

Geometric adstock with decay θ:
```
A(x_t) = x_t + θ × A(x_{t-1})
```
Decay values by channel:
- TV       : θ ∈ [0.3, 0.7] — long memory
- Facebook : θ ∈ [0.1, 0.5] — medium
- Search   : θ ∈ [0.0, 0.3] — short (intent-driven)
- OOH      : θ ∈ [0.2, 0.6] — medium
- Print    : θ ∈ [0.1, 0.4] — short-medium

## 3. Saturation (Diminishing Returns)

Hill transformation:
```
Hill(x) = x^S / (K^S + x^S)
```
- K = half-saturation point (spend at which effect = 50% of max)
- S = shape (S > 1 → sigmoidal, S < 1 → concave)

## 4. Bayesian Inference

### Priors
```
β_i    ~ HalfNormal(σ=1)    # channel coefficients
θ_i    ~ Beta(2, 2)          # adstock decay ∈ [0,1]
K_i    ~ Gamma(3, 1)         # half-saturation
S_i    ~ Gamma(2, 1)         # Hill shape
σ      ~ HalfNormal(0.5)     # observation noise
```

### Likelihood
```
Revenue(t) ~ Normal(μ(t), σ)
```

### Sampling
- Algorithm : NUTS (No-U-Turn Sampler)
- Draws     : 1 000 per chain
- Tune      : 1 000 (warmup)
- Chains    : 4
- target_accept : 0.9

## 5. Convergence Diagnostics

| Metric | Threshold | Description                      |
|--------|-----------|----------------------------------|
| R-hat  | < 1.01    | Gelman-Rubin convergence         |
| ESS    | > 400     | Effective Sample Size            |
| LOO-CV | —         | Leave-One-Out Cross Validation   |

## 6. ROI Calculation

```
ROI_channel = ΣRevenue_contribution_channel / ΣSpend_channel
```
Reported with 94% HDI (Highest Density Interval).

## References

- Jin et al. (2017) — "Bayesian Methods for Media Mix Modeling"
- Google Meridian — https://developers.google.com/meridian
- Meta Robyn — https://facebookexperimental.github.io/Robyn/
- PyMC-Marketing — https://www.pymc-marketing.io/
