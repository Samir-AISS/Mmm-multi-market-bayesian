# Data Dictionary — MMM Multi-Market

## Dataset principal

**Fichier :** `data/synthetic/mmm_multi_market.csv`  
**Dimensions :** 2 080 lignes × 14 colonnes  
**Granularité :** 1 ligne = 1 marché × 1 semaine  
**Période :** 2020-01-06 → 2023-12-25 (208 semaines × 10 marchés)  
**Seed :** 42 (reproductible)

---

## Description des colonnes

| Colonne            | Type    | Unité   | Description                                      |
|--------------------|---------|---------|--------------------------------------------------|
| `market`           | string  | —       | Code pays ISO : FR, DE, UK, IT, ES, NL, BE, PL, SE, NO |
| `date`             | date    | —       | Date du lundi de la semaine (format YYYY-MM-DD)  |
| `week`             | int     | —       | Numéro de semaine [1, 208]                       |
| `revenue`          | float   | €       | Revenus hebdomadaires (variable cible)           |
| `tv_spend`         | float   | €       | Dépenses TV de la semaine                        |
| `facebook_spend`   | float   | €       | Dépenses publicité Facebook/Meta                 |
| `search_spend`     | float   | €       | Dépenses Search (Google Ads, Bing)               |
| `ooh_spend`        | float   | €       | Dépenses Out-Of-Home (affichage)                 |
| `print_spend`      | float   | €       | Dépenses Print (presse, magazines)               |
| `competitor_price` | float   | index   | Prix concurrents (base 100)                      |
| `events`           | int     | 0/1     | Événement exceptionnel cette semaine (binaire)   |
| `trend`            | float   | ratio   | Tendance long terme multiplicative [0.5, 2.0]    |
| `seasonality`      | float   | ratio   | Saisonnalité hebdomadaire multiplicative [0.5, 2.0] |
| `promotions`       | int     | 0/1     | Promotion active cette semaine (binaire)         |

---

## Marchés disponibles

| Code | Pays        | Base Revenue  | Type saisonnalité | Trend annuel |
|------|-------------|---------------|-------------------|--------------|
| FR   | France      | 500 000 €     | Standard          | +2.0%        |
| DE   | Allemagne   | 620 000 €     | Standard          | +1.5%        |
| UK   | Royaume-Uni | 580 000 €     | Mild              | +2.5%        |
| IT   | Italie      | 430 000 €     | Mediterranean     | +1.0%        |
| ES   | Espagne     | 390 000 €     | Mediterranean     | +1.8%        |
| NL   | Pays-Bas    | 340 000 €     | Mild              | +2.2%        |
| BE   | Belgique    | 310 000 €     | Standard          | +1.2%        |
| PL   | Pologne     | 280 000 €     | Eastern           | +3.5%        |
| SE   | Suède       | 360 000 €     | Nordic            | +2.0%        |
| NO   | Norvège     | 380 000 €     | Nordic            | +1.8%        |

---

## Canaux marketing

| Canal      | Part budget | Decay θ | ROI typique | Effet mémoire |
|------------|-------------|---------|-------------|---------------|
| TV         | 40%         | 0.60    | 1.5 – 2.0x  | Long          |
| Facebook   | 20%         | 0.30    | 1.2 – 1.8x  | Moyen         |
| Search     | 20%         | 0.15    | 2.0 – 3.0x  | Court         |
| OOH        | 12%         | 0.45    | 0.8 – 1.2x  | Moyen         |
| Print      | 8%          | 0.25    | 0.5 – 1.0x  | Court-Moyen   |

---

## Statistiques descriptives (France, 208 semaines)

| Variable         | Min         | Médiane     | Max         | Std         |
|------------------|-------------|-------------|-------------|-------------|
| revenue          | ~300 000 €  | ~520 000 €  | ~900 000 €  | ~100 000 €  |
| tv_spend         | ~80 000 €   | ~170 000 €  | ~400 000 €  | ~60 000 €   |
| facebook_spend   | ~30 000 €   | ~80 000 €   | ~200 000 €  | ~30 000 €   |
| search_spend     | ~25 000 €   | ~75 000 €   | ~180 000 €  | ~28 000 €   |
| seasonality      | 0.75        | 1.00        | 1.40        | 0.15        |
| trend            | 1.00        | 1.08        | 1.17        | 0.05        |

---

## Contraintes de qualité

- Aucune valeur manquante (`NaN`)
- Revenue > 0 pour toutes les lignes
- Spend ≥ 0 pour tous les canaux
- Events et Promotions ∈ {0, 1}
- Semaines consécutives [1, 208] par marché
- Aucun doublon sur (market, week)