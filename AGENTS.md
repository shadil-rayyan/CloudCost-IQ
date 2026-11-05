# AGENTS.md — CloudCost IQ

**Document Version:** 2.0  
**Project:** CloudCost IQ  
**Development Approach:** Incremental MVP releases  
**Primary Language:** Python  
**ML Training Environment:** Google Colab  
**Deployment Approach:** Local development first, containerized deployment later

---

# 1. Project Identity

**Project Name:** CloudCost IQ  
**Project Type:** Multi-Cloud Cost Intelligence and Prediction Platform  
**Cloud Providers:** AWS, Microsoft Azure, Google Cloud Platform  
**Application Type:** Web-based analytics dashboard  
**Primary Users:** Developers, cloud engineers, students, startups, and small engineering teams.

## Product Description

CloudCost IQ is a multi-cloud FinOps platform that enables users to understand, analyze, predict, and optimize their cloud spending.

The application accepts billing exports from Amazon Web Services, Microsoft Azure, and Google Cloud Platform. It validates and normalizes provider-specific billing information into a common data structure.

The platform provides:

- Unified multi-cloud cost visibility.
- Provider and account comparisons.
- Service, project, resource, and region cost analysis.
- Historical spending analysis.
- Machine-learning-based future cost prediction.
- Forecast uncertainty estimates.
- Cost anomaly detection.
- Explainable optimization recommendations.
- Budget monitoring.
- Billing data quality monitoring.
- Model performance and training information.
- Import history and downloadable reports.

The product should progress through three releases:

**MVP 1:** Complete multi-cloud billing analytics application.

**MVP 2:** Advanced machine-learning forecasting and intelligence.

**MVP 3:** Production MLOps, automated cloud billing connections, monitoring, and operational hardening.

The application must never display fabricated ML predictions as real results.

---

# 2. Instructions for the AI Coding Agent

You are the primary software development agent responsible for implementing CloudCost IQ.

Follow these instructions whenever creating, modifying, or refactoring project files.

## Core Development Rules

1. Use Python throughout the initial application.
2. Do not introduce Rust.
3. Do not introduce unnecessary microservices.
4. Keep the architecture modular and maintainable.
5. Build functional features instead of decorative placeholder screens.
6. Keep ingestion, processing, analytics, ML, and UI logic separate.
7. Preserve original billing information during normalization.
8. Never fabricate real billing data, predictions, ML metrics, or potential savings.
9. Synthetic data is allowed only for explicitly labelled demonstrations and automated tests.
10. Do not silently discard invalid billing records.
11. Do not combine incompatible currencies or cost bases.
12. Write automated tests for important business logic.
13. Document important functions and configuration parameters.
14. Never commit credentials, private billing data, or sensitive artifacts.
15. Prefer simple, reliable solutions over unnecessary infrastructure.
16. Do not implement all three MVP releases simultaneously.
17. Complete and test each MVP before moving to the next.
18. Do not mark a feature complete merely because its UI exists.
19. All dashboard pages must display real application state or a useful empty state.
20. An ML model must pass evaluation checks before being used by the application.

## Feature Development Workflow

Before implementing a feature:

1. Inspect the existing repository.
2. Understand current architecture and data structures.
3. Identify dependencies and affected modules.
4. Reuse existing code wherever appropriate.
5. Implement the business logic.
6. Write tests.
7. Integrate the UI or API.
8. Verify compatibility with existing features.
9. Update relevant documentation.

Do not rewrite a working module without a clear technical reason.

---

# 3. Product Scope and MVP Releases

The project must be developed incrementally.

## 3.1 MVP 1 — Multi-Cloud Cost Intelligence Foundation

**Objective:** Deliver a complete and usable multi-cloud cost analytics platform.

MVP 1 must work independently without requiring a trained ML model.

### Required features

| Module | Features |
|---|---|
| Application foundation | Streamlit application, FastAPI, database connections, configuration |
| Billing uploads | AWS, Azure, GCP, and supported FOCUS files |
| File validation | Format, schema, dates, currencies, and cost validation |
| Normalization | Common multi-cloud billing schema |
| Storage | Original file storage, Parquet, DuckDB, SQLite |
| Overview dashboard | Spending KPIs, provider comparison, service breakdown |
| Cost analysis | Account, project, provider, service, region, and date filters |
| Import history | Upload history and processing status |
| Data quality | Coverage, duplicates, invalid records, missing periods |
| Reporting | CSV exports and cost summaries |
| Recommendations | Basic evidence-based spending recommendations |
| Settings | Reporting preferences, currency configuration, and workspace settings |

### Dashboard requirements for MVP 1

All main dashboard pages must be functional:

- Overview.
- Import Billing Data.
- Cost Analysis.
- ML Predictions.
- Anomaly Detection.
- Cost Optimization.
- Import History.
- Data Quality.
- Settings.

The Forecast page must show a clear message such as:

"Forecasting is not yet available. A validated production model has not been registered."

It must not show randomly generated predictions or sample model performance.

The anomaly and recommendation pages may initially use transparent statistical or rule-based logic. They must not pretend that an advanced ML model has been deployed.

### MVP 1 Definition of Done

A user must be able to:

1. Open the application.
2. Upload AWS billing data.
3. Upload Azure billing data.
4. Upload GCP billing data.
5. Inspect and validate imported records.
6. Analyze combined and provider-specific costs.
7. Filter spending.
8. Inspect services and accounts.
9. View import history.
10. Understand data-quality limitations.
11. Export reports.

MVP 1 is complete only when the entire process works using real billing exports.

---

## 3.2 MVP 2 — Advanced ML Cost Prediction

**Objective:** Implement one serious forecasting architecture trained in Google Colab and integrated into CloudCost IQ.

This is the primary advanced machine-learning component of the academic project.

### Model decision

Use:

**Global Multi-Horizon XGBoost Quantile Regression.**

Use one XGBoost model architecture with multi-quantile output.

The model must provide:

- P10: lower forecast estimate.
- P50: median forecast estimate.
- P90: upper forecast estimate.

These values provide an estimated forecast interval rather than displaying a single unsupported cost prediction.

Use XGBoost's `reg:quantileerror` objective with multiple quantile values.

Recommended initial quantiles:

`[0.10, 0.50, 0.90]`

Use a supported XGBoost version of at least 2.0.

The model must use histogram-based tree construction:

`tree_method = hist`

GPU training may be enabled in Colab when available, but CPU training must remain supported.

### Why this architecture?

Cloud billing is structured numerical and categorical data.

Historical spending can be influenced by:

- Recent cost trends.
- Weekly patterns.
- Monthly patterns.
- Provider.
- Cloud service.
- Account or project.
- Region.
- Usage changes.
- Billing-period timing.

A global gradient-boosted model with time-series feature engineering is a practical first production-oriented architecture for this data.

Do not add LSTM, GRU, Prophet, ARIMA, RandomForest, LightGBM, or additional forecasting models to MVP 2.

Historical-average and seasonal-naive methods may be used as evaluation benchmarks. They do not count as additional ML models.

Do not claim XGBoost is universally the best forecasting algorithm. Its suitability must be demonstrated through out-of-sample evaluation.

### MVP 2 Features

| Module | Required functionality |
|---|---|
| Dataset preparation | Real billing data to daily time-series dataset |
| Feature engineering | Lag, rolling, calendar, horizon, and categorical features |
| ML training | Global XGBoost multi-quantile model |
| Validation | Rolling-origin time-series backtesting |
| Evaluation | MAE, RMSE, WAPE, pinball loss, interval coverage |
| Explainability | Feature importance and SHAP-based analysis where supported |
| Model artifact | Versioned model and preprocessing artifacts |
| Colab integration | Reproducible training notebook |
| MLflow | Experiment and artifact tracking |
| Forecast API | Load and execute a validated model |
| Forecast dashboard | Historical versus predicted costs and uncertainty |
| Anomalies | Detect unusual spending relative to expected cost behavior |
| Recommendations | Evidence-based cost optimization suggestions |

### MVP 2 Definition of Done

Users must be able to:

1. Upload real multi-cloud billing data.
2. Prepare eligible daily cost time series.
3. Train the XGBoost model through Google Colab.
4. Evaluate it against time-series baselines.
5. Export its artifacts.
6. Integrate the model into CloudCost IQ.
7. Generate 30-day predictions.
8. See P10, P50, and P90 forecasts.
9. Inspect historical model performance.
10. Understand prediction uncertainty.
11. Investigate unusual spending.
12. View explainable recommendations.

Predictions must be based on actual historical billing data.

---

## 3.3 MVP 3 — Production MLOps and Cloud Integration

**Objective:** Improve the application from a working ML project into a production-oriented platform.

### Features

- Model registry.
- Model version management.
- Champion model selection.
- Model promotion and rollback.
- Data drift monitoring.
- Prediction error monitoring.
- Scheduled retraining.
- Model performance dashboards.
- Automated billing imports.
- Direct AWS integration.
- Direct Azure integration.
- Direct GCP BigQuery integration.
- Cloud telemetry integrations.
- Improved workspace security.
- Multi-user access management.
- Deployment monitoring.
- CI/CD automation.

### MVP 3 Definition of Done

The application must support a controlled model lifecycle:

```text
DATA COLLECTION
       |
       v
DATA VALIDATION
       |
       v
MODEL TRAINING
       |
       v
TIME-SERIES EVALUATION
       |
       v
MODEL VALIDATION GATES
       |
       v
MODEL REGISTRY
       |
       v
MODEL PROMOTION
       |
       v
PRODUCTION INFERENCE
       |
       v
PERFORMANCE MONITORING
       |
       v
RETRAINING OR ROLLBACK
```

Do not implement autonomous cloud-resource modification as part of this release.

---

# 4. Technology Stack

Use the following stack unless a demonstrated technical requirement requires a change.

| Component | Technology | Purpose |
|---|---|---|
| Main language | Python 3.11+ | Application and ML |
| Frontend | Streamlit | Dashboard |
| Backend | FastAPI | REST API |
| Data processing | Pandas, NumPy | Data preparation |
| File support | PyArrow | Parquet processing |
| Analytics | DuckDB | Analytical queries |
| Metadata | SQLite | Import and application metadata |
| Data storage | Parquet | Processed billing records |
| Main ML model | XGBoost | Multi-quantile forecasting |
| ML preparation | Scikit-learn | Preprocessing |
| Explainability | SHAP | Model interpretation |
| Visualization | Plotly | Interactive charts |
| Model artifacts | XGBoost native model format | Model serialization |
| Preprocessing artifacts | Joblib | Trusted preprocessing serialization |
| Experiment tracking | MLflow | Training and evaluation tracking |
| Testing | Pytest | Automated tests |
| Configuration | Pydantic Settings | Application configuration |
| Containers | Docker | Reproducible environment |
| Local orchestration | Docker Compose | Application services |
| CI/CD | GitHub Actions | Automated checks |
| Monitoring | Prometheus and Grafana | Later-stage monitoring |

### Dependency rules

- Keep application dependencies and ML training dependencies organized.
- Do not install unnecessary packages.
- Pin or constrain versions for reproducibility.
- Record installed library versions in every training run.
- Ensure the training environment and inference environment use compatible XGBoost versions.

---

# 5. Application Architecture

Use a modular monolithic architecture initially.

```text
                         USER
                           |
                           v
                      STREAMLIT UI
                           |
                           v
                        FASTAPI
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
          INGESTION     ANALYTICS        ML
             |             |             |
             v             v             v
        VALIDATION      DUCKDB       PREDICTION
             |             |             |
             v             v             v
        NORMALIZATION   PARQUET     MODEL ARTIFACT
             |                           |
             +-------------+-------------+
                           |
                           v
                  RECOMMENDATION ENGINE
                           |
                           v
                       DASHBOARD
```

### Architectural rules

- Provider-specific parsers must be independent.
- Normalization must not contain dashboard logic.
- Analytics must not depend on Streamlit.
- Model training must not happen inside a dashboard request.
- Inference must use the same feature definitions as training.
- ML artifacts must be stored separately from application source code.
- Use API services for business operations.
- Keep the architecture simple enough to run on a developer's laptop.

Google Colab is a training environment, not a runtime dependency of the application.

---

# 6. Project Folder Structure

Maintain this project structure.

```text
cloudcost-iq/
│
├── AGENTS.md
├── README.md
├── requirements.txt
├── requirements-ml.txt
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── configs/
│   ├── app_config.yaml
│   └── ml_config.yaml
│
├── app/
│   ├── main.py
│   │
│   ├── pages/
│   │   ├── overview.py
│   │   ├── upload.py
│   │   ├── cost_analysis.py
│   │   ├── forecast.py
│   │   ├── anomalies.py
│   │   ├── recommendations.py
│   │   ├── import_history.py
│   │   ├── data_quality.py
│   │   └── settings.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   └── routes/
│   │       ├── imports.py
│   │       ├── costs.py
│   │       ├── forecasts.py
│   │       ├── anomalies.py
│   │       └── recommendations.py
│   │
│   ├── ingestion/
│   │   ├── base_parser.py
│   │   ├── aws_parser.py
│   │   ├── azure_parser.py
│   │   ├── gcp_parser.py
│   │   ├── focus_parser.py
│   │   ├── file_detector.py
│   │   └── import_manager.py
│   │
│   ├── processing/
│   │   ├── validation.py
│   │   ├── normalization.py
│   │   ├── deduplication.py
│   │   ├── aggregation.py
│   │   ├── currency.py
│   │   └── data_quality.py
│   │
│   ├── analytics/
│   │   ├── cost_summary.py
│   │   ├── cost_breakdown.py
│   │   ├── cost_trends.py
│   │   └── provider_comparison.py
│   │
│   ├── ml/
│   │   ├── dataset_builder.py
│   │   ├── feature_engineering.py
│   │   ├── time_series_split.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── explain.py
│   │   ├── artifact_manager.py
│   │   └── model_contract.py
│   │
│   ├── recommendations/
│   │   ├── rules.py
│   │   ├── recommendation_engine.py
│   │   └── recommendation_status.py
│   │
│   ├── database/
│   │   ├── connection.py
│   │   ├── models.py
│   │   └── repositories.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── constants.py
│   │
│   └── utils/
│       ├── dates.py
│       ├── files.py
│       └── formatting.py
│
├── scripts/
│   ├── prepare_training_data.py
│   ├── train_model.py
│   ├── validate_model.py
│   └── inspect_dataset.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── rejected/
│   ├── demo/
│   ├── reference/
│   └── metadata/
│
├── artifacts/
│   └── models/
│       ├── candidates/
│       ├── staging/
│       └── production/
│
├── notebooks/
│   └── colab_training.ipynb
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── ingestion/
│   ├── analytics/
│   ├── ml/
│   └── security/
│
├── mlruns/
│
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    ├── dataset_sources.md
    ├── ingestion_guide.md
    ├── ml_training_guide.md
    ├── model_card.md
    ├── ml_pipeline.md
    └── user_guide.md
```

### Folder rules

- `data/raw/`: original billing imports.
- `data/processed/`: normalized billing records.
- `data/rejected/`: rejected files and records, subject to retention controls.
- `data/demo/`: synthetic or demonstration data.
- `data/reference/`: approved public reference datasets.
- `artifacts/models/candidates/`: newly trained model candidates.
- `artifacts/models/staging/`: models being validated.
- `artifacts/models/production/`: approved inference artifacts.
- `notebooks/`: Google Colab and research notebooks.
- `mlruns/`: MLflow local tracking data.
- `docs/`: technical and user documentation.

Never commit private billing data, `.env` files, model credentials, or large training artifacts.

---

# 7. Dataset Sources and Selection

The AI coding agent must distinguish between:

1. Billing datasets used to test ingestion.
2. Billing datasets used to validate dashboards.
3. Billing datasets used to train forecasting models.
4. Synthetic datasets used only for test and demonstration purposes.

Do not treat these as interchangeable.

## 7.1 Primary Multi-Cloud Reference Dataset: FOCUS Sandbox

Use the FinOps Foundation FOCUS Sandbox as the first public multi-cloud reference dataset.

Purpose:

- Understand FOCUS billing fields.
- Test multi-cloud normalization.
- Validate provider identification.
- Test service-level cost breakdowns.
- Test dashboards.
- Test SQL analytical queries.
- Develop ingestion integration tests.

The documented FOCUS Sandbox contains anonymized real AWS and GCP billing data. Its Azure data uses Microsoft sample data combined with real data.

The documented sandbox example uses data from September 2024.

Therefore:

**Do not use the FOCUS Sandbox as the only forecasting training dataset.**

It is a valuable ingestion and dashboard reference, but its documented temporal coverage is insufficient by itself for reliable 30-day forecasting.

## 7.2 AWS Dataset

Preferred source:

**AWS Cost and Usage Report 2.0 through AWS Data Exports.**

Use actual billing data exported from an AWS account.

Preferred data granularity:

- Daily.
- Hourly, when available and useful.

Support:

- CSV.
- Parquet.
- Supported compressed exports.

Account for:

- Payer account.
- Linked account.
- Service.
- SKU.
- Resource ID, when available.
- Region.
- Usage period.
- Invoice period.
- Tags.
- Credits.
- Discounts.
- Commitment-related charges.
- Actual and amortized cost information.

Preserve AWS-specific fields.

Do not assume every billing record includes a resource ID.

## 7.3 Azure Dataset

Preferred source:

**Azure Cost Management Exports.**

Support:

- Actual Cost exports.
- Amortized Cost exports.
- FOCUS exports.
- CSV.
- Parquet.

Account for:

- Subscription.
- Resource group.
- Resource ID.
- Service.
- Meter.
- SKU.
- Location.
- Usage period.
- Invoice information.
- Tags.
- Credits.
- Currency.
- Cost type.

Keep Actual and Amortized costs separate.

Microsoft FinOps Toolkit open data may be used for understanding Azure schema fields, pricing units, and region mappings.

Its demonstration exports must not be assumed to be production billing histories.

## 7.4 GCP Dataset

Preferred source:

**Google Cloud Detailed Cloud Billing Export to BigQuery.**

For the initial application, users may download the billing data as CSV or Parquet and upload it.

Account for:

- Billing account.
- Project.
- Service.
- SKU.
- Resource ID, where available.
- Region.
- Usage start and end times.
- Invoice month.
- Credits.
- Currency.
- Cost adjustments.

Use the detailed export for resource-level cost analysis when available.

A direct BigQuery integration is planned for MVP 3.

## 7.5 FOCUS Support

Support FOCUS billing files as a unified input format.

At the time of implementation, verify the provider's actual FOCUS schema version.

Do not hardcode the assumption that all providers export identical versions.

The parser must:

- Identify the FOCUS version.
- Validate required columns.
- Map standard fields.
- Preserve provider-specific extension fields.
- Store schema version metadata.
- Reject unsupported incompatible versions with understandable errors.

## 7.6 Dataset Hierarchy

Use the following dataset-selection rules:

| Dataset | Ingestion testing | Dashboard testing | ML training |
|---|---|---|---|
| FOCUS Sandbox | Yes | Yes | Limited research only |
| AWS CUR 2.0 real export | Yes | Yes | Yes |
| Azure real Cost Management export | Yes | Yes | Yes |
| GCP real Detailed Billing export | Yes | Yes | Yes |
| Microsoft sample export | Schema testing | Limited | No |
| Synthetic test billing data | Yes | Demo only | No production claims |

### Important dataset principle

Do not search for random datasets simply to increase the row count.

The training data must represent cloud billing behavior and must contain meaningful time information and cost values.

A dataset containing only pricing information, resource metrics, or synthetic values is not an equivalent replacement for historical billing data.

---

# 8. Billing Ingestion Workflow

Implement:

```text
OPEN IMPORT PAGE
       |
       v
SELECT CLOUD PROVIDER
       |
       v
SELECT OR DETECT FORMAT
       |
       v
UPLOAD BILLING FILE
       |
       v
INSPECT FILE AND SCHEMA
       |
       v
VALIDATE BILLING RECORDS
       |
       v
SHOW DATA PREVIEW
       |
       v
SHOW WARNINGS AND ERRORS
       |
       v
USER CONFIRMS IMPORT
       |
       v
NORMALIZE BILLING DATA
       |
       v
SAVE PROCESSED DATA
       |
       v
CREATE IMPORT SUMMARY
```

### Supported uploads

Initially support:

- CSV.
- Parquet.
- GZIP-compressed supported files.

ZIP can be supported with safe archive extraction controls.

Validate actual file contents, not just the file extension.

Protect against:

- Oversized uploads.
- Corrupt files.
- Decompression bombs.
- Path traversal.
- Unsupported schemas.
- Unexpected memory consumption.

Do not execute uploaded content.

### Import tracking

Every import must store:

- Import ID.
- Provider.
- Filename.
- File size.
- Format.
- Upload time.
- Processing status.
- Schema version.
- Date coverage.
- Cost basis.
- Currency.
- Records read.
- Valid records.
- Rejected records.
- Duplicate information.
- Validation warnings.
- Error information.

### Reimport behavior

Billing exports can change after their original delivery.

Implement:

- File fingerprinting.
- Duplicate import detection.
- Import lineage.
- Controlled replacement of billing periods.
- Reprocessing of revised files.
- Record identity where provider data supports it.

Do not deduplicate billing records just because their visible columns happen to be identical.

Do not silently combine overlapping exports.

---

# 9. Normalized Billing Data Model

Create a unified structure supporting:

| Field | Description |
|---|---|
| `record_id` | Internal unique record ID |
| `provider` | AWS, Azure, GCP, or FOCUS |
| `billing_account_id` | Billing account ID |
| `billing_account_name` | Account name |
| `linked_account_id` | AWS linked account |
| `subscription_id` | Azure subscription |
| `project_id` | GCP project |
| `project_name` | Project name |
| `service_name` | Cloud service |
| `service_category` | Normalized service group, where possible |
| `sku_id` | Product or SKU |
| `sku_description` | SKU description |
| `resource_id` | Resource ID, when available |
| `region` | Region |
| `usage_start` | Usage start timestamp |
| `usage_end` | Usage end timestamp |
| `charge_period_start` | Charge period start |
| `charge_period_end` | Charge period end |
| `invoice_month` | Invoice month |
| `cost_amount` | Normalized cost |
| `currency` | Original currency |
| `cost_basis` | Billed, actual, amortized, effective, or supported source basis |
| `quantity` | Usage quantity |
| `unit` | Unit of measurement |
| `credits_amount` | Credits |
| `tax_amount` | Tax, where separately available |
| `charge_category` | Usage, purchase, tax, credit, adjustment, etc. |
| `charge_type` | Provider-specific charge type |
| `pricing_model` | On-demand, reservation, savings plan, etc. |
| `tags_json` | Billing or resource tags |
| `raw_fields_json` | Original provider fields |
| `source_file_id` | Source file |
| `import_batch_id` | Import batch |
| `schema_version` | Normalized schema version |

Unavailable fields must remain null.

Never invent missing values.

## Cost semantics

- Keep actual and amortized cost separate.
- Preserve negative credits and refunds.
- Preserve invoice month independently from usage date.
- Preserve original currency.
- Keep billed cost separate from effective or amortized cost.
- Do not silently remove adjustments.
- Do not sum incompatible currencies.

For invoice-spending forecasts, use a clearly documented billed-cost target.

For consumption-oriented analysis, effective or amortized cost can be analyzed separately.

The initial ML forecast must use one clearly defined target cost basis.

---

# 10. Data Quality Requirements

Data quality is a visible product feature.

Validate:

- Missing required fields.
- Invalid cost values.
- Invalid dates.
- Missing currencies.
- Unsupported schema versions.
- Incorrect usage periods.
- Corrupt files.
- Duplicate imports.
- Overlapping coverage.
- Missing resource IDs.
- Negative credits and adjustments.
- Incomplete periods.
- Unexpected data distributions.

Do not classify every missing resource ID or tag as an error.

### Missing days and zero spending

A missing billing day does not automatically mean that the cost was zero.

Distinguish:

- Complete day with positive cost.
- Complete day with zero cost.
- Incomplete day.
- Missing day.
- Billing data not yet available.

Only create zero-cost observations when source coverage establishes that the period is complete and no billable records were present.

The ML dataset builder must use these quality indicators.

### Freshness

Display:

- Last uploaded date.
- Latest usage date.
- Latest invoice month.
- Import completion time.
- Date coverage.
- Incomplete periods.
- Known billing-data delays.

Do not describe delayed billing data as real-time information.

---

# 11. Dashboard Requirements

The dashboard must be fully usable in MVP 1.

## 11.1 Overview

Display:

- Total spending, where compatible.
- AWS cost.
- Azure cost.
- GCP cost.
- Historical cost trend.
- Provider comparison.
- Top expensive services.
- Spending distribution.
- Recent alerts.
- Forecast status.
- Recommendation summary.
- Data freshness.

Never show a misleading combined monetary total when currencies or cost bases are incompatible.

## 11.2 Cost Analysis

Provide:

- Provider filters.
- Account filters.
- Subscription filters.
- Project filters.
- Service filters.
- Region filters.
- Date filters.
- Cost trend charts.
- Provider comparisons.
- Service breakdowns.
- Account breakdowns.
- Detailed billing tables.

## 11.3 Forecast Page

In MVP 1, show model readiness and data coverage.

In MVP 2, display:

- Historical cost.
- P10 predicted cost.
- P50 predicted cost.
- P90 predicted cost.
- Forecast horizon.
- Currency.
- Cost basis.
- Model version.
- Model training date.
- Historical evaluation metrics.
- Forecast uncertainty.
- Provider-level forecast.
- Account and service-level forecast, where eligible.

Display a clear explanation when the model cannot produce a forecast.

## 11.4 Anomaly Page

Display:

- Detection date.
- Provider.
- Account.
- Project.
- Service.
- Expected cost.
- Observed cost.
- Percentage difference.
- Severity.
- Detection method.
- Explanation.
- Link to cost details.

## 11.5 Recommendations

Each recommendation must contain:

- Recommendation ID.
- Title.
- Provider.
- Affected account or project.
- Service.
- Observation.
- Evidence.
- Suggested investigation.
- Risk or limitation.
- Estimated savings, only where justified.
- Priority.
- Status.
- Creation date.

Recommendation statuses:

- New.
- Under Review.
- Accepted.
- Rejected.
- Completed.

---

# 12. Advanced ML Forecasting Specification

This section defines the actual ML implementation.

The coding agent must follow this design unless a technical limitation is identified and documented.

## 12.1 Forecasting Objective

Predict the next 30 days of cloud spending using historical daily billing records.

The forecast must support:

- Selected cloud provider.
- Compatible billing account.
- Compatible currency.
- Compatible cost basis.
- Eligible service-level groups.
- Combined provider-level reporting when compatible.

### Forecast target

Use daily aggregated billed cost as the initial model target.

Do not train directly on arbitrary raw billing line items.

First aggregate the normalized records into daily time series.

### Time field

Use the charge period or usage timestamp according to the normalized provider mapping.

Preserve invoice month separately.

The daily aggregation timezone must be documented and consistent.

### Forecast granularity

Initial forecast frequency:

**Daily.**

Initial prediction horizon:

**30 days.**

Monthly-only billing datasets are not eligible for daily forecasting.

Do not manufacture daily values from monthly totals.

---

## 12.2 Training Series Definition

Create panel time-series data.

Initial series identity:

```text
provider
+
billing_account_id
+
service_name
+
currency
+
cost_basis
```

Resource-level forecasting is not required for MVP 2 because resource-level data can be sparse and inconsistent.

Project and region can be added to the series identity where sufficient history exists.

Use a global model across eligible series within a compatible currency and cost-basis cohort.

The model must learn common spending behavior across multiple cloud accounts and providers while retaining provider and service information as features.

### Currency handling

Never train on incompatible raw currency values as though they represented the same monetary unit.

For MVP 2:

- Keep native reporting currencies separate.
- Train within compatible currency groups.
- Use one compatible cost basis per model artifact.
- Do not sum incompatible currency forecasts.
- Do not perform undocumented foreign exchange conversion.

Future reporting-currency conversion requires a documented, date-aware FX source.

---

## 12.3 Required ML Dataset Eligibility

The dataset builder must produce a quality report before training.

Recommended initial requirements:

| Requirement | Initial rule |
|---|---|
| Forecast frequency | Daily |
| Forecast horizon | 30 days |
| Preferred historical coverage | At least 365 days |
| Minimum historical coverage | 180 calendar days |
| Minimum observed daily records per forecast series | Configurable; target at least 120 |
| Billing completeness | Incomplete dates excluded or explicitly handled |
| Currency | One compatible currency per training cohort |
| Cost basis | One compatible cost basis per training cohort |
| Target | Daily billed cost |
| Input data | Real normalized billing exports |

These are initial engineering quality gates, not universal mathematical guarantees.

A dataset must also support at least three valid rolling-origin evaluation windows.

If the required history or evaluation structure is unavailable:

- Do not manufacture extra history.
- Do not claim the model is production-ready.
- Display a data sufficiency warning.
- Continue allowing the rest of the application to function.

Prefer 12 months or more of real billing history.

---

## 12.4 Feature Engineering

Create features using only information available at the forecast origin.

### Historical lag features

Start with:

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

These represent previous daily cost observations.

### Rolling statistics

Create:

- 7-day rolling mean.
- 7-day rolling median.
- 7-day rolling standard deviation.
- 28-day rolling mean.
- 28-day rolling median.
- 28-day rolling standard deviation.
- 7-day rolling sum.
- 28-day rolling sum.

### Trend features

Create:

- Short-term cost change.
- Long-term cost change.
- Recent versus historical average.
- Exponentially weighted moving average, where appropriate.

### Calendar features

Create:

- Day of week.
- Weekend indicator.
- Day of month.
- Week of year.
- Month.
- Quarter.
- Target-date calendar features.

### Forecast-specific features

Create:

- Forecast horizon.
- Target date.
- Provider.
- Service category.
- Service name.
- Region, when useful.
- Account or project grouping, where suitable.

Do not use high-cardinality resource IDs as ordinary categorical features without a clear justification.

### Feature leakage prevention

All historical features must be calculated from data available before the forecast origin.

Do not use future target costs in:

- Rolling averages.
- Normalization.
- Imputation.
- Feature selection.
- Feature encoding.
- Model validation.
- Hyperparameter tuning.

Add automated leakage tests.

---

## 12.5 Forecasting Strategy

Use a global direct multi-horizon supervised forecasting strategy.

For every eligible time series:

1. Select a historical forecast origin.
2. Calculate lag and rolling features using records available at that origin.
3. Create examples for forecast horizons 1 through 30.
4. Add the horizon as an input feature.
5. Use future calendar features for the target date.
6. Use actual future daily cost as the training label.
7. Train across all eligible series.

The resulting model should learn relationships across providers and services.

At inference time, calculate the current historical features once and create inputs for the next 30 forecast dates.

The model should predict each horizon directly.

Do not use recursive predictions as lag values for later forecast days in MVP 2.

This avoids the automatic accumulation of recursive forecast errors.

---

## 12.6 Model Training

Use one XGBoost multi-quantile model.

Core configuration:

```python
MODEL_CONFIG = {
    "model_name": "cloudcost_iq_global_xgboost_quantile",
    "objective": "reg:quantileerror",
    "quantile_alpha": [0.10, 0.50, 0.90],
    "tree_method": "hist",
    "forecast_horizon_days": 30,
    "random_state": 42,
}
```

The agent may tune suitable hyperparameters using earlier time-based validation windows.

Candidate hyperparameters include:

- Maximum tree depth.
- Learning rate.
- Number of boosting rounds.
- Minimum child weight.
- Subsample.
- Column sampling.
- L1 regularization.
- L2 regularization.

Use early stopping against a chronological validation set.

Do not randomly split time-series rows.

Do not use the final test period for hyperparameter tuning.

Avoid blindly copying hyperparameters from unrelated datasets.

Save the final selected configuration.

### Quantile crossing

XGBoost documents that independently estimated quantiles can cross.

The inference pipeline must:

1. Inspect the ordering of P10, P50, and P90.
2. Calculate and record the quantile crossing rate.
3. Apply a documented monotonicity correction or reject an invalid output.
4. Confirm that P10 <= P50 <= P90 before displaying forecasts.
5. Monitor the frequency of corrections.

Do not silently ignore quantile crossing.

---

## 12.7 Time-Series Model Validation

Use rolling-origin or expanding-window backtesting.

Illustrative design:

```text
FOLD 1
TRAIN HISTORY | VALIDATION: NEXT 30 DAYS

FOLD 2
       TRAIN HISTORY | VALIDATION: NEXT 30 DAYS

FOLD 3
              TRAIN HISTORY | VALIDATION: NEXT 30 DAYS

FINAL
                     TRAIN/VALIDATION | UNTOUCHED TEST
```

The exact dates must depend on the dataset's available coverage.

Rules:

- Training observations must precede validation observations.
- Forecast target dates must not leak across a split.
- Each fold must simulate the intended 30-day production forecast.
- Feature construction must use only information available at each origin.
- The final test period must remain untouched during model selection.
- Report performance across multiple forecast origins.

Do not use random train-test splitting for forecasting.

---

## 12.8 Evaluation Metrics

Calculate:

### Point forecast metrics

- MAE.
- RMSE.
- WAPE.
- MASE where the baseline definition is appropriate.

Avoid relying on MAPE when actual costs may be zero or close to zero.

### Quantile forecast metrics

- Pinball loss for P10.
- Pinball loss for P50.
- Pinball loss for P90.

### Uncertainty evaluation

- P10–P90 interval coverage.
- Average prediction interval width.
- Quantile crossing percentage.

For a nominal P10–P90 interval, measure whether approximately 80% of actual observations fall inside the interval.

Do not assume that nominal coverage automatically means actual coverage is calibrated.

### Report metrics separately

Provide metrics by:

- Overall dataset.
- Cloud provider.
- Currency.
- Forecast horizon bucket.
- Eligible service group.

Where useful, report both aggregate-weighted and per-series average performance.

### Baseline comparison

Compare the XGBoost model with simple reference methods:

- Trailing 7-day average.
- Seasonal-naive weekly forecast.

These are benchmarks, not additional ML architectures.

Only promote the XGBoost model when it meets documented evaluation requirements and demonstrates useful performance relative to the selected baseline.

Do not invent an accuracy percentage or claim improvement without actual backtesting.

---

# 13. Google Colab Training Workflow

Google Colab is the designated environment for model experiments and training.

The notebook must be:

`notebooks/colab_training.ipynb`

The notebook should be reproducible and use real normalized billing data.

Do not put all production feature engineering and training code directly into notebook cells.

The notebook should import reusable functions from the application ML modules.

This ensures that training and inference use the same feature definitions.

## 13.1 Colab Notebook Sections

### Section 1 — Environment Setup

- Install compatible dependencies.
- Record Python version.
- Record XGBoost version.
- Record Pandas, NumPy, and Scikit-learn versions.
- Record execution environment.
- Set random seed.
- Configure experiment tracking.

### Section 2 — Load the Project

Load the CloudCost IQ repository from its documented Git commit or release.

Install the project's ML dependencies.

Do not allow the notebook to silently use an unknown version of the code.

### Section 3 — Load Billing Dataset

Load normalized billing Parquet data from Google Drive or an approved storage location.

Also support a FOCUS-format input path for dataset preparation experiments.

Do not hardcode private account identifiers or credentials.

### Section 4 — Dataset Validation

Check:

- Required columns.
- Schema version.
- Supported currencies.
- Cost basis.
- Date coverage.
- Null values.
- Duplicate information.
- Data completeness.
- Valid cost values.
- Number of eligible series.

Generate a dataset-quality report.

### Section 5 — Cost Aggregation

- Select the approved cost field.
- Filter the correct cost basis.
- Group records into daily costs.
- Build the time-series panel.
- Preserve provider and account information.
- Track missing and incomplete periods.
- Do not silently turn missing billing information into zero.

### Section 6 — Exploratory Data Analysis

Generate:

- Daily cost trends.
- Service cost distributions.
- Provider cost distributions.
- Time-series coverage.
- Missing-data analysis.
- Cost outlier analysis.
- Currency distribution.
- Dataset quality summary.

### Section 7 — Feature Engineering

Use the reusable feature-engineering module.

Generate:

- Historical lags.
- Rolling statistics.
- Trend features.
- Calendar features.
- Forecast horizon.
- Provider and service categorical features.

Verify that no future data is used.

### Section 8 — Backtesting Dataset

Create chronological rolling-origin folds.

Generate an untouched final test period.

Display the exact date ranges used in:

- Training.
- Validation.
- Final testing.

### Section 9 — Train XGBoost

Train the global multi-quantile XGBoost model.

Use:

- P10.
- P50.
- P90.

Use early stopping based on chronological validation.

Capture all relevant parameters.

### Section 10 — Evaluate Model

Calculate:

- MAE.
- RMSE.
- WAPE.
- Pinball loss.
- Interval coverage.
- Interval width.
- Quantile crossing rate.

Compare against the historical-average and seasonal-naive baselines.

Produce charts of actual versus predicted cost.

### Section 11 — Model Explainability

Generate:

- Global feature importance.
- SHAP summary information where supported.
- Feature contribution analysis.
- Important provider and service effects.

Do not interpret feature importance as proof of causation.

### Section 12 — Final Model Training

After model selection and final test reporting:

- Fit the selected model using the approved training data.
- Keep the previously measured out-of-sample evaluation results.
- Record the training data date range.
- Record the final model configuration.
- Create the production candidate artifact.

### Section 13 — Model Validation

Run:

- Model loading test.
- Feature schema validation.
- Prediction shape validation.
- Quantile ordering validation.
- Prediction range checks.
- Reproducibility checks.
- Inference smoke test.

### Section 14 — Save Model

Export all required artifacts.

### Section 15 — Download or Transfer Artifacts

Save the model bundle to Google Drive or download it into the CloudCost IQ repository's local artifact directory.

Do not depend on the Colab session remaining active.

---

# 14. Model Artifact Contract

A trained model cannot be integrated into the application as an unexplained `.pkl` file.

Every model must have a documented artifact bundle.

Recommended structure:

```text
artifacts/models/candidates/<model_version>/
│
├── model.ubj
├── preprocessor.joblib
├── feature_schema.json
├── model_config.json
├── metrics.json
├── training_manifest.json
├── dataset_quality.json
└── model_card.md
```

### Artifact requirements

#### `model.ubj`

The trained XGBoost model in a supported native model format.

#### `preprocessor.joblib`

Trusted fitted preprocessing artifact, if preprocessing is required.

Never load untrusted serialized artifacts.

#### `feature_schema.json`

Contains:

- Feature names.
- Feature types.
- Feature order.
- Categorical mappings.
- Feature schema version.

#### `model_config.json`

Contains:

- Model name.
- Forecast horizon.
- Quantile configuration.
- Model parameters.
- Target definition.
- Cost basis.
- Currency.
- Training configuration.

#### `metrics.json`

Contains all model evaluation metrics.

Include baseline comparison metrics.

#### `training_manifest.json`

Contains:

- Model version.
- Dataset fingerprint.
- Dataset date range.
- Training timestamp.
- Code Git commit.
- Package versions.
- Random seed.
- Provider coverage.
- Currency.
- Cost basis.
- Feature schema version.
- Artifact checksums.

#### `dataset_quality.json`

Contains:

- Total records.
- Eligible records.
- Rejected records.
- Missing periods.
- Number of time series.
- Coverage information.
- Dataset sufficiency assessment.

#### `model_card.md`

Explain:

- Model purpose.
- Training dataset.
- Forecast target.
- Forecast granularity.
- Feature engineering.
- Model architecture.
- Evaluation methodology.
- Performance metrics.
- Known limitations.
- Intended use.
- Limitations on business decisions.

A model without the required metadata must not be promoted.

---

# 15. MLflow and Experiment Tracking

Use MLflow for experiment tracking in MVP 2.

Track:

- Dataset fingerprint.
- Code version.
- Training date.
- Model parameters.
- Feature definitions.
- Forecast horizon.
- Quantile configuration.
- Rolling-origin fold metrics.
- Final test metrics.
- Model artifacts.
- Dataset-quality report.
- Evaluation charts.

Store artifacts securely.

Do not upload private raw billing datasets to a public MLflow server.

In Colab, use a local or approved remote tracking location. Preserve experiment results outside the temporary runtime.

Do not confuse MLflow tracking with a completed production model registry.

Full registry-based model lifecycle management is an MVP 3 feature.

---

# 16. Model Inference and Dashboard Integration

The deployed application must load an approved model artifact.

Inference workflow:

```text
USER SELECTS FORECAST SCOPE
          |
          v
LOAD NORMALIZED BILLING DATA
          |
          v
CHECK DATA ELIGIBILITY
          |
          v
CHECK CURRENCY AND COST BASIS
          |
          v
LOAD APPROVED MODEL ARTIFACT
          |
          v
VALIDATE FEATURE SCHEMA
          |
          v
GENERATE NEXT 30 DAYS OF FEATURES
          |
          v
RUN XGBOOST INFERENCE
          |
          v
VALIDATE P10 / P50 / P90
          |
          v
SAVE FORECAST RESULTS
          |
          v
DISPLAY FORECAST DASHBOARD
```

### Inference requirements

- Do not train a model during normal dashboard page loads.
- Do not silently switch models.
- Validate model compatibility.
- Validate selected currency.
- Validate cost basis.
- Validate historical coverage.
- Check forecast horizon.
- Check quantile ordering.
- Display the model version.
- Display prediction timestamps.
- Record inference status.

### Insufficient history

If the data is insufficient, display a clear message.

Do not generate fictional future costs.

The cost-analysis dashboard must continue working even when ML predictions are unavailable.

---

# 17. Anomaly Detection

Implement interpretable anomaly detection first.

In MVP 1:

- Detect large cost changes using transparent statistical rules.
- Identify significant differences from recent spending.
- Flag incomplete billing periods.

In MVP 2:

- Use out-of-sample forecast errors and prediction intervals to identify unusual costs.
- Compare actual spending against the expected forecast distribution.
- Use P10 and P90 as reference bounds where the model has been validated.
- Identify large cost deviations.
- Separate billing-data problems from possible infrastructure spending problems.

Do not identify an anomaly solely from an in-sample model prediction.

Do not describe a statistical anomaly as a confirmed infrastructure failure.

---

# 18. Optimization Recommendation Engine

Use an explainable rule-based recommendation engine.

Example rules:

| Condition | Recommendation |
|---|---|
| Repeated service-level cost increase | Investigate usage and deployment changes |
| Spending exceeds configured budget | Review projected month-end spending |
| Sudden spending increase | Investigate the affected service and period |
| Increasing storage expenditure | Review storage growth and retention policies |
| Large spending concentration | Review service allocation |
| Provider recommendation available | Display the original recommendation |

Every recommendation must contain:

- Required data fields.
- Detection logic.
- Supporting evidence.
- Recommendation text.
- Risk.
- Priority.
- Known limitations.

### CPU and memory limitations

Billing exports do not reliably provide CPU and memory utilization.

Never recommend machine rightsizing based only on billing cost.

Future telemetry integrations:

- AWS CloudWatch.
- Azure Monitor.
- Google Cloud Monitoring.

Keep telemetry data and billing data as separate but joinable sources.

### Infrastructure safety

The application must not automatically:

- Delete resources.
- Shut down resources.
- Resize resources.
- Modify cloud configurations.
- Purchase cloud commitments.

All recommendations require human review.

---

# 19. API Requirements

Create versioned REST endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/imports` | Upload billing files |
| GET | `/api/v1/imports` | List imports |
| GET | `/api/v1/imports/{id}` | Import details |
| GET | `/api/v1/costs/summary` | Cost summary |
| GET | `/api/v1/costs/timeseries` | Cost time series |
| GET | `/api/v1/costs/breakdown` | Cost breakdown |
| POST | `/api/v1/forecasts` | Generate forecast |
| GET | `/api/v1/forecasts/{id}` | Get forecast |
| GET | `/api/v1/anomalies` | List anomalies |
| GET | `/api/v1/recommendations` | List recommendations |
| PATCH | `/api/v1/recommendations/{id}` | Update recommendation |
| GET | `/api/v1/health` | Health check |

Use Pydantic request and response schemas.

Return structured errors.

Do not expose internal file paths, credentials, raw server errors, or sensitive billing records.

---

# 20. DevOps and Production Engineering

## 20.1 Docker

Provide:

- Dockerfile.
- Docker Compose configuration.
- Persistent storage.
- Environment configuration.
- Health checks.

The local application should be launchable through documented commands.

Do not introduce Kubernetes for the MVP.

## 20.2 CI/CD

Use GitHub Actions for:

- Dependency checks.
- Linting.
- Unit tests.
- Ingestion tests.
- API tests.
- ML inference tests.
- Docker builds.

Do not make CI dependent on private cloud billing files.

## 20.3 ML Validation Gate

A candidate model must pass the following checks before production use:

1. Artifact exists and can be loaded.
2. Model version is recorded.
3. Feature schema is compatible.
4. Dataset fingerprint is available.
5. Forecast horizon is correct.
6. Evaluation results are present.
7. Model performance is compared against baselines.
8. Quantile order is valid.
9. Inference smoke test passes.
10. Known limitations are documented.

A model must not be automatically promoted merely because training completed successfully.

## 20.4 Monitoring

In MVP 3, monitor:

- Import failures.
- Data freshness.
- Dataset drift.
- Feature drift.
- Prediction failures.
- Model inference latency.
- Forecast interval coverage.
- Actual-versus-predicted error.
- Model artifact loading.
- Model version in production.

Do not log private billing details in monitoring labels.

## 20.5 Model Retraining

Future retraining workflow:

1. New billing data arrives.
2. Data quality checks run.
3. Training eligibility is evaluated.
4. Candidate model is trained.
5. Rolling backtesting is performed.
6. Existing production model is compared.
7. Candidate model passes promotion gates.
8. Model version is registered.
9. Approved model is promoted.
10. Previous production model remains available for rollback.

Do not overwrite production artifacts without versioning.

---

# 21. Security Requirements

Billing files may contain commercially sensitive information.

Implement:

- File size limits.
- Safe file parsing.
- Upload isolation.
- Secure temporary files.
- Path traversal protection.
- Safe archive extraction.
- Environment-based configuration.
- Restricted data access.
- Input validation.
- Safe logging.
- Appropriate error handling.

Never:

- Hardcode cloud credentials.
- Commit `.env` files.
- Commit customer billing data.
- Store credentials in billing tables.
- Execute uploaded content.
- Expose raw billing exports through public URLs.
- Store private credentials inside notebooks.
- Share Colab notebooks with sensitive output cells.

For MVP 3 direct cloud connections, prefer short-lived or federated credentials and read-only permissions.

The MVP must work with billing-file uploads without requiring cloud credentials.

---

# 22. Testing Requirements

## Ingestion

Test:

- AWS CUR 2.0.
- Azure Cost Management exports.
- GCP Detailed Billing exports.
- FOCUS files.
- CSV parsing.
- Parquet parsing.
- GZIP support.
- Invalid schemas.
- Empty files.
- Invalid dates.
- Missing mandatory fields.
- Repeated imports.
- Multiple currencies.
- Large files.

## Normalization

Test:

- Provider mappings.
- Cost basis.
- Negative credits.
- Refunds.
- Tax fields.
- Resource IDs.
- Tags.
- Invoice month.
- Usage timestamps.
- Missing optional fields.
- Source lineage.

## Analytics

Test:

- Provider comparison.
- Service breakdown.
- Account breakdown.
- Date filters.
- Currency handling.
- Cost-basis separation.
- Incomplete billing periods.
- Empty data.
- Export generation.

## ML dataset construction

Test:

- Correct daily aggregation.
- Currency separation.
- Cost-basis separation.
- Missing period handling.
- Genuine zero-cost days.
- Time ordering.
- Lag feature calculations.
- Rolling feature calculations.
- Forecast horizon creation.
- Series construction.
- No future-data leakage.

## ML training and evaluation

Test:

- Chronological splits.
- Rolling-origin folds.
- Baseline calculations.
- XGBoost training.
- Multi-quantile prediction output.
- Quantile crossing detection.
- MAE.
- RMSE.
- WAPE.
- Pinball loss.
- Interval coverage.
- Artifact creation.
- Model loading.

## ML inference

Test:

- Model feature compatibility.
- Missing features.
- Wrong currency.
- Wrong cost basis.
- Insufficient history.
- Prediction horizon.
- Output shape.
- Valid quantile ordering.
- Inference reproducibility within documented tolerance.

## Security

Test:

- Invalid files.
- Oversized files.
- Unsafe ZIP files.
- Path traversal.
- Invalid API input.
- Sensitive error exposure.

---

# 23. Development Plan

Follow this exact order.

## MVP 1 — Complete Cost Analytics Product

Build:

1. Project foundation.
2. Configuration.
3. Streamlit application.
4. FastAPI backend.
5. SQLite metadata.
6. DuckDB analytics.
7. AWS parser.
8. Azure parser.
9. GCP parser.
10. FOCUS parser.
11. Billing validation.
12. Normalization.
13. Parquet storage.
14. Import history.
15. Data quality.
16. Overview dashboard.
17. Cost analysis.
18. Basic recommendations.
19. Reports.
20. Automated tests.

Deliverable:

A fully functional multi-cloud billing analytics application using real billing exports.

No fake ML predictions.

## MVP 2 — Advanced Machine Learning

Build:

1. Real training dataset preparation.
2. Daily time-series aggregation.
3. Data sufficiency checks.
4. Feature engineering.
5. Global multi-horizon dataset.
6. XGBoost multi-quantile model.
7. Rolling-origin backtesting.
8. Baseline comparisons.
9. Model evaluation.
10. Explainability.
11. Google Colab training notebook.
12. MLflow experiment tracking.
13. Model artifact generation.
14. Artifact validation.
15. Prediction API.
16. Forecast dashboard.
17. Forecast-based anomaly detection.
18. Explainable recommendation integration.
19. ML tests.

Deliverable:

A working application with reproducible training, validated predictions, prediction intervals, and model-performance reporting.

## MVP 3 — Production MLOps and Automation

Build:

1. Model registry.
2. Model versioning.
3. Promotion and rollback.
4. Data drift detection.
5. Performance monitoring.
6. Scheduled retraining.
7. Automated cloud billing ingestion.
8. AWS integration.
9. Azure integration.
10. GCP BigQuery integration.
11. Cloud telemetry.
12. Improved access management.
13. Monitoring dashboards.
14. Deployment pipeline.

Deliverable:

A production-oriented multi-cloud FinOps system with a managed ML lifecycle.

---

# 24. README Requirements

Maintain a complete README containing:

1. Product introduction.
2. Features.
3. Architecture.
4. Technology stack.
5. MVP roadmap.
6. Installation.
7. Configuration.
8. Docker startup.
9. AWS billing import.
10. Azure billing import.
11. GCP billing import.
12. Supported FOCUS versions.
13. Data model.
14. Google Colab training instructions.
15. Dataset requirements.
16. Feature engineering explanation.
17. XGBoost model description.
18. Evaluation methodology.
19. Model artifact integration.
20. Forecast dashboard instructions.
21. Known limitations.
22. Testing instructions.
23. Future enhancements.

The README must not claim features are implemented when they are only planned.

---

# 25. Definition of Done

## MVP 1

- All three providers can upload supported billing files.
- Billing data is validated and normalized.
- Original data is preserved.
- Cost analytics work.
- Currency handling is correct.
- Cost-basis handling is correct.
- All dashboard pages have meaningful behavior.
- Import history works.
- Data quality is visible.
- Reports can be exported.
- Tests pass.

## MVP 2

- The application uses real historical billing data for actual ML predictions.
- Dataset eligibility is validated.
- One global XGBoost multi-quantile architecture is implemented.
- The model forecasts 30 days.
- Training is reproducible in Google Colab.
- Feature engineering is reusable outside the notebook.
- Time-series evaluation is chronological.
- Baseline comparisons are available.
- Metrics are calculated from out-of-sample predictions.
- P10, P50, and P90 are produced.
- Prediction uncertainty is displayed.
- Quantile crossing is checked.
- Model artifacts include metadata.
- The application loads approved artifacts.
- Insufficient history is handled safely.
- MLflow experiment tracking works.
- Explainability results are available.
- No fake ML values are shown.

## MVP 3

- Model versions are registered.
- Candidate models are validated.
- Model promotion and rollback work.
- Drift and performance monitoring work.
- Retraining is controlled.
- Direct cloud integrations use appropriate permissions.
- Billing data updates are monitored.
- Deployment checks are automated.

---

# 26. Final Product and ML Principle

CloudCost IQ must follow this sequence:

**Correct billing data → Reliable normalization → Useful analytics → Reproducible ML → Validated predictions → Explainable recommendations → Monitored production model.**

The project must demonstrate both software engineering and advanced machine learning.

Do not build a dashboard that only looks convincing.

Do not train a model on a short dataset and claim production accuracy.

Do not use random train-test splitting for time-series forecasting.

Do not generate daily cost values from monthly billing information.

Do not mix currencies or incompatible cost definitions.

Do not display ML predictions unless a validated model artifact is available.

A successful project is one in which the full data pipeline, advanced ML training workflow, inference integration, and dashboard operate together.

**Build MVP 1 completely before moving to MVP 2. Complete and validate MVP 2 before attempting MVP 3.**
