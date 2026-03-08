# Methodology — MMM Multi-Market Bayesian

## 1. Objectif

Ce projet implémente un **Marketing Mix Model (MMM) bayésien multi-marchés**
inspiré de Google Meridian et Meta Robyn. Il mesure l'impact de 5 canaux
marketing sur les revenus de 10 marchés européens, et fournit des ROI
avec intervalles de crédibilité pour guider l'allocation budgétaire.

---

## 2. Équation du modèle
```
Revenue(t) ~ Normal(μ(t), σ)

μ(t) = BaseSales
     + Σᵢ [ βᵢ × Hill(Adstock(Spendᵢ(t), θᵢ), Kᵢ, Sᵢ) ]
     + γ_season × Seasonality(t)
     + γ_trend  × Trend(t)
     + γ_events × Events(t)
```

---

## 3. Transformation Adstock (Carryover Effect)

L'adstock capture l'effet mémoriel de la publicité : une campagne TV
de cette semaine influence encore les ventes des semaines suivantes.

**Adstock géométrique :**
```
A[t] = spend[t] + θ × A[t-1]
```

**Demi-vie :** `t₁/₂ = -log(2) / log(θ)`

| Canal     | Decay θ      | Demi-vie approx. |
|-----------|-------------|-----------------|
| TV        | 0.30 – 0.70 | 1 – 2 semaines  |
| Facebook  | 0.10 – 0.50 | < 1 semaine     |
| Search    | 0.00 – 0.30 | Immédiat        |
| OOH       | 0.20 – 0.60 | 1 semaine       |
| Print     | 0.10 – 0.40 | < 1 semaine     |

---

## 4. Saturation Hill (Rendements décroissants)

Au-delà d'un certain seuil de dépenses, l'efficacité marginale diminue.

**Formule Hill :**
```
Hill(x) = x^S / (K^S + x^S)
```

| Paramètre | Description                        | Valeur typique |
|-----------|------------------------------------|---------------|
| K         | Half-saturation point (Hill(K)=0.5)| Médiane spend |
| S         | Shape (S>1 → sigmoïde, S<1 → concave) | 1.5 – 3.0  |

**Propriétés :**
- `Hill(0)   = 0`
- `Hill(K)   = 0.5`
- `Hill(∞)   → 1`

---

## 5. Inférence bayésienne

### Priors
```python
# Coefficients canaux (strictement positifs)
β_i    ~ HalfNormal(σ=1.0)

# Decay adstock
θ_i    ~ Beta(α=2, β=2)        # ∈ [0, 1], centré sur 0.5

# Hill half-saturation
K_i    ~ Gamma(α=3, β=1)

# Hill shape
S_i    ~ Gamma(α=2, β=1)       # > 0

# Effets de contrôle
γ_*    ~ Normal(μ=0, σ=0.5)

# Bruit d'observation
σ      ~ HalfNormal(σ=0.5)
```

### Likelihood
```
Revenue(t) ~ Normal(μ(t), σ)
```

### Sampling MCMC (NUTS)
| Paramètre      | Valeur |
|----------------|--------|
| Algorithme     | NUTS (No-U-Turn Sampler) |
| Draws          | 1 000 par chaîne |
| Tune (warmup)  | 1 000 |
| Chaînes        | 4     |
| target_accept  | 0.90  |
| random_seed    | 42    |

---

## 6. Diagnostics de convergence

| Métrique   | Seuil     | Interprétation                        |
|------------|-----------|---------------------------------------|
| R-hat      | < 1.01    | Gelman-Rubin — convergence des chaînes|
| ESS bulk   | > 400     | Effective Sample Size                 |
| ESS tail   | > 400     | Qualité des queues de distribution    |
| Divergences| = 0       | Aucun problème géométrique HMC        |
| LOO-CV     | max elpd  | Leave-One-Out Cross Validation        |

---

## 7. Calcul du ROI
```
ROI_canal = Σ(contribution_canal) / Σ(spend_canal)
```

Rapporté avec intervalle de crédibilité HDI 94% :
```python
roi_samples = contribution_samples / total_spend
roi_lower   = np.percentile(roi_samples, 3)
roi_upper   = np.percentile(roi_samples, 97)
```

---

## 8. Validation du modèle

1. **Walk-Forward Validation** : split temporel glissant (5 folds)
2. **Posterior Predictive Check (PPC)** : couverture HDI 94% > 80%
3. **Métriques prédictives** : R² > 0.85, MAPE < 15%, NRMSE < 0.15
4. **Cohérence cross-market** : détection des marchés outliers

---

## 9. Références

- Jin et al. (2017) — *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects*
- Google Meridian — https://developers.google.com/meridian
- Meta Robyn — https://facebookexperimental.github.io/Robyn/
- PyMC-Marketing — https://www.pymc-marketing.io/
- Gelman & Rubin (1992) — *Inference from Iterative Simulation Using Multiple Sequences*