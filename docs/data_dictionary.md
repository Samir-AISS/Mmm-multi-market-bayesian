# Data Dictionary — MMM Multi-Market Dataset

## Dataset Overview
- **Rows**: 2 080 (10 markets × 208 weeks)
- **Columns**: 14
- **Period**: 2020-01-06 to 2023-12-25 (approx. 4 years)
- **Source**: Synthetic data inspired by Meta Robyn demo dataset

## Variables

| Column            | Type    | Description                              | Constraints              |
|-------------------|---------|------------------------------------------|--------------------------|
| market            | str     | Market ISO code (FR, DE, UK…)            | 10 exact values          |
| date              | str     | Week start date (Monday)                 | 2020-01-06 to 2023-12-25 |
| week              | int     | Week number (1 to 208)                   | Unique per market        |
| revenue           | float   | Weekly revenue (€)                       | > 80 000 €               |
| tv_spend          | float   | TV advertising spend (€)                 | > print_spend            |
| facebook_spend    | float   | Facebook/Meta spend (€)                  | ≥ 0                      |
| search_spend      | float   | Paid search spend (€)                    | ≥ 0                      |
| ooh_spend         | float   | Out-of-Home advertising spend (€)        | ≥ 0                      |
| print_spend       | float   | Print advertising spend (€)              | < tv_spend               |
| competitor_price  | float   | Competitor price index                   | ∈ [0.85, 1.15]           |
| events            | int     | Major market event this week             | ∈ {0, 1}                 |
| trend             | float   | Long-term growth trend                   | ∈ [0.0, 0.15]            |
| seasonality       | float   | Seasonal multiplier                      | ∈ [0.7, 1.5]             |
| promotions        | int     | Promotion active this week               | ∈ {0, 1}                 |

## Markets

| Code | Country     | GDP Index | Digital Maturity | Seasonality   |
|------|-------------|-----------|------------------|---------------|
| FR   | France      | 1.00      | 0.85             | Standard      |
| DE   | Germany     | 1.15      | 0.90             | Standard      |
| UK   | UK          | 1.10      | 0.95             | Mild          |
| IT   | Italy       | 0.85      | 0.75             | Mediterranean |
| ES   | Spain       | 0.80      | 0.78             | Mediterranean |
| NL   | Netherlands | 1.25      | 0.98             | Standard      |
| BE   | Belgium     | 1.05      | 0.88             | Standard      |
| PL   | Poland      | 0.65      | 0.82             | Eastern       |
| SE   | Sweden      | 1.30      | 0.99             | Nordic        |
| NO   | Norway      | 1.50      | 0.97             | Nordic        |
