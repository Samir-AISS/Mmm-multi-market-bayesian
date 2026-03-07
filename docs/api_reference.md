# API Reference

## src.data.multi_market_generator

### `generate_full_dataset() → pd.DataFrame`
Génère le dataset complet 2 080 lignes.

### `generate_market_data(market, params, rng) → pd.DataFrame`
Génère 208 semaines pour un marché.

---

## src.data.data_validator

### `validate(df) → ValidationReport`
Lance les 4 niveaux de validation. Retourne un rapport.

### `ValidationReport.print_summary()`
Affiche le rapport coloré avec PASS/FAIL par test.

---

## src.models.bayesian_mmm

### `BayesianMMM(config, market)`
Modèle MMM bayésien PyMC.

### `.build_model(df)` — TODO
### `.fit(df, draws, tune, chains)` — TODO
### `.predict(df)` — TODO
### `.get_contributions(df)` — TODO
### `.get_roi()` — TODO

---

## src.evaluation.metrics

### `r_squared(y_true, y_pred)` — TODO
### `mape(y_true, y_pred)` — TODO
### `nrmse(y_true, y_pred)` — TODO
