# AGENTS.md — CloudCost IQ

## 1. Project Identity

**Project Name:** CloudCost IQ  
**Project Type:** Multi-Cloud Cost Intelligence and Prediction Platform  
**Primary Language:** Python  
**Cloud Providers:** AWS, Microsoft Azure, Google Cloud Platform  
**Application Type:** Web-based analytics dashboard  
**Primary Users:** Developers, cloud engineers, students, startups, and small engineering teams.

### Product Description

CloudCost IQ is a multi-cloud FinOps platform that helps users understand, analyze, predict, and optimize their cloud spending.

Users can upload billing exports from AWS, Azure, and GCP into one centralized workspace. The application validates and normalizes provider-specific billing information into a common data structure.

Users can then:

- View cloud spending in a unified dashboard.
- Compare AWS, Azure, and GCP costs.
- Analyze spending by account, project, service, region, and time period.
- Forecast future cloud spending using machine learning.
- Detect unusual cost increases and spending patterns.
- Receive explainable, evidence-based cost optimization recommendations.
- Track recommendations and monitor their outcomes.
- Review billing data quality and import history.

The initial application is a **cost intelligence and decision-support platform**. It does not automatically modify cloud infrastructure.

---

# 2. Instructions for the AI Coding Agent

You are the primary software development agent responsible for building CloudCost IQ.

Follow these instructions whenever you create, edit, or refactor project files.

### Core development rules

1. Build functional features, not decorative placeholder screens.
2. Use Python for the entire initial application.
3. Do not introduce Rust.
4. Do not create unnecessary microservices.
5. Keep the architecture modular and easy to maintain.
6. Separate data ingestion, processing, analytics, machine learning, and presentation logic.
7. Preserve original billing information during normalization.
8. Never fabricate real billing data, model predictions, accuracy scores, or potential savings.
9. Synthetic data is permitted only for clearly labelled demo and test environments.
10. Never silently ignore invalid billing records.
11. Never mix incompatible currencies or cost bases into a misleading total.
12. Write tests for important data processing and business logic.
13. Document important functions, modules, and configuration options.
14. Do not commit cloud credentials, real billing data, or other sensitive information.
15. Prefer simple, reliable implementations over unnecessary complexity.

### How to approach new features

Before implementing a feature:

1. Inspect the existing project structure.
2. Understand the current data model and service boundaries.
3. Identify the files that need modification.
4. Reuse existing utilities where possible.
5. Add appropriate tests.
6. Update the documentation when behavior changes.
7. Verify that existing features continue to work.

Do not replace an existing working implementation simply because another implementation is more fashionable.

---

# 3. Product Scope

## 3.1 MVP Features

The first working version must implement:

| Module | Required functionality | Priority |
|---|---|---|
| Billing ingestion | Upload AWS, Azure, GCP billing files | Critical |
| Data validation | Validate schema, fields, dates, currency, and cost values | Critical |
| Normalization | Convert provider billing records into a common schema | Critical |
| Data storage | Store processed records in Parquet and metadata in SQLite | Critical |
| Overview dashboard | Display total and provider-specific costs | Critical |
| Cost analysis | Filter costs by provider, account, project, service, and date | Critical |
| Forecasting | Generate future cost estimates using validated ML models | High |
| Anomaly detection | Identify unexpected cost changes | High |
| Recommendations | Generate explainable cost optimization suggestions | High |
| Data quality | Display import history, coverage, and validation warnings | High |
| Reporting | Export selected reports | Medium |
| Model tracking | Track model versions and evaluation results | Medium |
| Direct cloud connections | Connect provider accounts using read-only permissions | Later |

## 3.2 Explicitly Out of Scope for MVP

Do not implement these unless specifically requested:

- Automatic shutdown of cloud resources.
- Automatic modification of virtual machine sizes.
- Purchasing or modifying Reserved Instances.
- Automatic purchasing of Savings Plans or commitments.
- Kubernetes autoscaling.
- Continuous live CPU and memory monitoring.
- Complex enterprise chargeback systems.
- Enterprise multi-tenant RBAC.
- Autonomous cloud infrastructure management.
- A conversational agent that executes cloud operations.

These capabilities require additional infrastructure data, permissions, validation, and safety mechanisms.

The MVP must focus on accurate billing analysis, forecasting, anomaly detection, and decision support.

---

# 4. Competitor-Informed Product Direction

Use established FinOps products as references for product behavior and interaction patterns.

Relevant competitors include:

- IBM Apptio Cloudability
- Broadcom CloudHealth
- Flexera One
- Finout
- CloudZero
- Vantage
- Harness Cloud Cost Management
- DoiT Cloud Intelligence
- Datadog Cloud Cost Management
- Ternary
- Amnic
- AWS Cost Explorer and Cost Optimization Hub
- Azure Cost Management
- Google Cloud FinOps Hub
- IBM Turbonomic
- CAST AI
- Kubecost
- OpenCost
- Infracost

Do not copy their proprietary designs or implementation.

Use the following general product principles:

| Competitor-inspired capability | How CloudCost IQ should implement it |
|---|---|
| Multi-cloud visibility | Unified AWS, Azure, and GCP cost dashboard |
| Cost allocation | Group spending by account, project, service, and tags |
| Forecasting | Predict future spending using historical billing data |
| Anomaly detection | Identify unexpected changes and explain them |
| Cost recommendations | Display evidence-based optimization opportunities |
| Data visualization | Provide interactive filters and detailed cost charts |
| FinOps reporting | Export understandable spending and forecast reports |
| Data transparency | Show import status, date coverage, and data-quality issues |

### Product differentiation

CloudCost IQ should prioritize:

1. Easy multi-cloud billing-file uploads.
2. Transparent data validation and normalization.
3. Explainable ML predictions.
4. Clear cost anomaly explanations.
5. Recommendations supported by actual billing evidence.
6. A straightforward interface suitable for smaller engineering teams.

Do not claim to be an enterprise-grade automated optimization platform.

---

# 5. Technology Stack

Use the following technology stack unless a technical limitation requires a change.

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python 3.11+ | Main application |
| Frontend | Streamlit | Interactive dashboard |
| Backend API | FastAPI | Application endpoints |
| Data processing | Pandas, NumPy | Cleaning and transformations |
| File processing | PyArrow | Parquet support |
| Analytics | DuckDB | Cost analysis and querying |
| Metadata storage | SQLite | Import and configuration metadata |
| Data storage | Parquet | Processed billing records |
| Machine learning | Scikit-learn | Forecasting and anomaly experiments |
| Optional ML | XGBoost | Alternative forecasting model |
| Visualization | Plotly | Interactive charts |
| Model serialization | Joblib | Model artifact storage |
| Experiment tracking | MLflow | Model experiment tracking |
| Testing | Pytest | Automated tests |
| API validation | Pydantic | Request and response validation |
| Containerization | Docker | Reproducible application environment |
| Orchestration | Docker Compose | Local service startup |
| CI/CD | GitHub Actions | Automated quality checks |
| Monitoring | Prometheus and Grafana | Later-stage application monitoring |

Avoid introducing new infrastructure unless it solves a demonstrated requirement.

---

# 6. Application Architecture

Use a modular architecture.

```text
                  USER
                    |
                    v
             STREAMLIT UI
                    |
                    v
               FASTAPI
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
    INGESTION   ANALYTICS       ML
        |           |           |
        v           v           v
   VALIDATION   DUCKDB       FORECASTING
        |           |           |
        v           v           v
  NORMALIZATION  PARQUET    ANOMALIES
        |                       |
        +-----------+-----------+
                    |
                    v
           RECOMMENDATION ENGINE
                    |
                    v
               DASHBOARD
```

### Architectural requirements

- Keep provider-specific parsing separate.
- Keep provider-independent analytics separate from ingestion.
- Keep the ML pipeline separate from the dashboard.
- Avoid placing application logic directly inside Streamlit page files.
- Use service modules that can be tested without running the UI.
- Use configuration files and environment variables where appropriate.
- Keep raw billing imports separate from normalized data.
- Keep model artifacts separate from source code.

---

# 7. Project Folder Structure

Create and maintain the following structure.

```text
cloudcost-iq/
│
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
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
│   │   ├── data_quality.py
│   │   ├── import_history.py
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
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── baselines.py
│   │   └── anomaly_detection.py
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
├── data/
│   ├── raw/
│   ├── processed/
│   ├── rejected/
│   ├── demo/
│   └── metadata/
│
├── artifacts/
│   └── models/
│       ├── staging/
│       └── production/
│
├── notebooks/
│   └── colab_training.ipynb
│
├── mlruns/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── ingestion/
│   ├── analytics/
│   └── ml/
│
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    ├── ingestion_guide.md
    ├── ml_pipeline.md
    └── user_guide.md
```

### Folder rules

- `data/raw/`: original uploaded files.
- `data/processed/`: normalized billing records.
- `data/rejected/`: rejected records or files, subject to safe retention rules.
- `data/demo/`: explicitly labelled demonstration data.
- `data/metadata/`: SQLite database and import metadata.
- `artifacts/models/`: trained model files.
- `notebooks/`: experiments and Google Colab training.
- `tests/`: automated tests.
- `docs/`: system and user documentation.

Never commit real billing exports or uploaded customer data to Git.

---

# 8. Billing Data Sources

Support real billing export data from all three cloud providers.

## 8.1 AWS

Preferred source:

**AWS Cost and Usage Report 2.0 through AWS Data Exports.**

Support the export formats implemented by the ingestion module, initially:

- CSV
- Parquet
- GZIP-compressed supported files

Account for:

- Payer account and linked accounts.
- Service and SKU information.
- Usage periods.
- Resource IDs, when provided.
- Regions.
- Tags.
- Credits and adjustments.
- Discounts and commitment-related charges.
- Invoice periods.
- Currency.
- Actual and amortized cost information, where available.

Do not assume every AWS export contains all resource-level information.

Where CUR 2.0 includes manifests, metadata, or partitioned files, process the export as a dataset rather than assuming that one file represents the complete billing period.

## 8.2 Microsoft Azure

Preferred source:

**Azure Cost Management Exports.**

Support the formats implemented by the ingestion module, initially:

- CSV
- Parquet

Account for:

- Subscription.
- Resource group.
- Resource ID, when present.
- Service name.
- Meter and SKU.
- Location.
- Usage date.
- Invoice information.
- Tags.
- Credits and adjustments.
- Currency.
- Actual and amortized cost exports.

Do not assume actual and amortized exports are interchangeable.

## 8.3 Google Cloud Platform

Preferred source:

**Detailed Cloud Billing Export to BigQuery.**

For the first version, allow users to download supported billing data into CSV or Parquet and upload it.

Account for:

- Billing account.
- Project ID.
- Project name.
- Service.
- SKU.
- Resource information, when available.
- Location.
- Usage start and end times.
- Invoice month.
- Credits.
- Currency.
- Cost adjustments.

A direct BigQuery integration may be introduced in a later version.

## 8.4 FOCUS

Support FOCUS-format billing files as an additional input option.

Use FOCUS to simplify common cross-provider reporting, but preserve provider-specific fields.

Do not assume FOCUS contains every attribute needed for detailed provider-specific optimization.

---

# 9. Billing Import Workflow

Implement the following user journey.

```text
OPEN IMPORT PAGE
       |
       v
SELECT PROVIDER
       |
       v
SELECT OR DETECT FILE FORMAT
       |
       v
UPLOAD BILLING FILES
       |
       v
INSPECT FILE STRUCTURE
       |
       v
DETECT SCHEMA
       |
       v
VALIDATE RECORDS
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

### Import page requirements

The interface must provide:

- Provider selection.
- File upload.
- Detected file format.
- Detected billing schema.
- Date-range preview.
- Currency information.
- Number of records.
- Invalid record count.
- Duplicate or reimport warnings.
- Import confirmation.
- Import progress and final status.

### Supported upload behavior

Initially support:

- CSV files.
- Parquet files.
- GZIP-compressed supported files.

ZIP support may be added with secure extraction controls.

Validate files by their actual contents and parser behavior, not only their filenames.

Do not execute uploaded content.

Protect against:

- Oversized files.
- Malformed files.
- Decompression bombs.
- ZIP path traversal.
- Unsupported schemas.
- Excessive memory usage.

### Import tracking

Every import must have an import identifier.

Record:

- Import ID.
- Provider.
- Original filename.
- File format.
- File size.
- Upload timestamp.
- Processing status.
- Date coverage.
- Number of records read.
- Number of valid records.
- Number of rejected records.
- Currency information.
- Cost basis.
- Warnings.
- Error information.
- Schema version.

### Reimports and billing revisions

Billing exports can be updated after an initial import.

The system must avoid accidentally double-counting the same billing dataset.

- Detect repeated files using file fingerprints.
- Use stable provider record identifiers where available.
- Preserve import history.
- Provide clear replace-period or reprocess behavior where necessary.
- Do not deduplicate legitimate billing records simply because their visible values are identical.
- Keep enough import lineage to explain which file produced each processed record.

Do not silently combine overlapping exports.

---

# 10. Normalized Billing Data Model

Create one common billing record structure for analysis.

The normalized table should support the following fields where available:

| Field | Description |
|---|---|
| `record_id` | Internal unique record identifier |
| `provider` | AWS, Azure, GCP, or FOCUS |
| `billing_account_id` | Billing account identifier |
| `billing_account_name` | Billing account name |
| `linked_account_id` | AWS linked account, when applicable |
| `subscription_id` | Azure subscription, when applicable |
| `project_id` | GCP project, when applicable |
| `project_name` | Project name |
| `service_name` | Cloud service |
| `sku_id` | Product or SKU identifier |
| `sku_description` | SKU description |
| `resource_id` | Resource identifier, if available |
| `region` | Resource or usage region |
| `usage_start` | Usage start timestamp |
| `usage_end` | Usage end timestamp |
| `invoice_month` | Billing invoice month |
| `cost_amount` | Selected normalized cost amount |
| `currency` | Original currency |
| `cost_basis` | Actual, amortized, or other supported basis |
| `quantity` | Usage quantity |
| `unit` | Usage unit |
| `credits_amount` | Credits, if available |
| `tax_amount` | Tax amount, if available |
| `charge_category` | Usage, purchase, tax, credit, adjustment, etc. |
| `charge_type` | Provider-specific charge type |
| `pricing_model` | On-demand, reservation, savings plan, etc. |
| `tags_json` | Resource or billing tags |
| `raw_fields_json` | Original provider fields |
| `source_file_id` | Original import source |
| `import_batch_id` | Import batch identifier |
| `schema_version` | Normalized schema version |

Fields that are unavailable must remain null or explicitly unavailable.

Do not invent missing resource IDs, tags, usage values, CPU metrics, memory metrics, or prices.

### Preserve original data

Always retain the original uploaded files or an appropriately managed raw copy.

Normalized records must retain their source lineage.

Provider-specific source columns should remain available through `raw_fields_json` or an equivalent preserved raw representation.

### Cost semantics

The application must:

- Keep actual and amortized costs separate.
- Preserve negative credits, refunds, and adjustments.
- Distinguish taxes from other charge categories when the source permits.
- Keep usage date separate from invoice month.
- Preserve the original source currency.
- Avoid combining incompatible cost bases.
- Avoid adding values from different currencies into one total.

If cost conversion is introduced later, require a documented exchange-rate source, exchange-rate date, and conversion methodology.

---

# 11. Data Quality and Reliability

Data quality is a core product feature, not just an internal process.

Validate:

- Missing mandatory columns.
- Invalid dates.
- Invalid numerical cost values.
- Missing currency.
- Unsupported provider.
- Unsupported billing schema.
- Invalid usage periods.
- Empty files.
- Corrupt files.
- Duplicate imports.
- Overlapping date ranges.
- Missing resource identifiers.
- Unexpected negative or zero values.
- Incomplete billing periods.

Not every missing field is an error. Resource IDs and tags, for example, may legitimately be absent.

### Missing data versus zero cost

A missing billing period is not automatically a zero-cost period.

Use explicit status information to distinguish:

- Complete period.
- Incomplete period.
- Missing period.
- Zero-cost period.
- Period with delayed billing records.

### Data freshness

Display:

- Last uploaded date.
- Latest usage date.
- Latest invoice month.
- Import completion time.
- Period coverage.
- Known data limitations.

Billing data can arrive late or be revised. Do not describe an import as real-time unless the implementation actually supports it.

---

# 12. Dashboard Requirements

Build a clean and consistent dashboard suitable for cloud engineers and small engineering teams.

Use readable layouts, restrained visual styling, helpful empty states, and consistent chart conventions.

## 12.1 Overview Dashboard

Display:

- Total cost, subject to currency and cost-basis compatibility.
- AWS cost.
- Azure cost.
- GCP cost.
- Historical cost trend.
- Provider comparison.
- Top expensive services.
- Spending distribution.
- Recent anomalies.
- Forecast summary.
- High-priority recommendations.
- Data freshness indicator.

Do not display a combined monetary total if the underlying data mixes currencies or incompatible cost bases.

## 12.2 Cost Analysis

Provide:

- Provider filter.
- Account filter.
- Project filter.
- Service filter.
- Region filter.
- Tag filter, when available.
- Date range selection.
- Cost trend chart.
- Service cost breakdown.
- Account comparison.
- Provider comparison.
- Detailed billing records.

Support drill-down where the source data contains the required dimensions.

## 12.3 Forecasting

Display:

- Historical spending.
- Predicted future spending.
- Forecast period.
- Model name.
- Training date.
- Evaluation metrics.
- Forecast uncertainty, where supported.
- Expected month-end cost when the model and data support it.

Never present an unsupported prediction as a guaranteed future bill.

## 12.4 Anomalies

Display:

- Detection date.
- Provider.
- Account or project.
- Affected service.
- Baseline cost.
- Observed cost.
- Absolute difference.
- Percentage difference.
- Detection method.
- Severity.
- Explanation.
- Link to cost details.

## 12.5 Recommendations

Every recommendation must have:

- Recommendation ID.
- Title.
- Provider.
- Affected account or project.
- Affected service.
- Observation.
- Supporting evidence.
- Suggested investigation or action.
- Risk or limitation.
- Estimated savings, only where defensibly calculated.
- Priority.
- Status.
- Creation date.

Recommendation statuses:

- New.
- Under Review.
- Accepted.
- Rejected.
- Completed.

Do not claim savings merely because spending increased or a resource appears expensive.

---

# 13. Machine Learning Requirements

The ML pipeline must use genuine billing data for actual application predictions.

Synthetic data must remain restricted to testing and explicitly labelled demonstrations.

## 13.1 Forecasting Pipeline

Implement:

1. Load normalized billing records.
2. Select a compatible provider, account, currency, and cost basis.
3. Aggregate cost by the available time granularity.
4. Check historical data coverage and quality.
5. Create time-series features.
6. Split data chronologically.
7. Train a baseline model.
8. Train an ML model.
9. Evaluate both models.
10. Save the selected model and its metadata.
11. Generate future predictions.
12. Present predictions and evaluation results.

### Candidate forecasting methods

Start with:

- Rolling average.
- Seasonal-naive baseline where suitable.
- RandomForest regression with lag features.

Other forecasting methods may be evaluated later.

Do not assume that RandomForest is automatically the best model.

### Time-series validation

Never randomly shuffle time-series records for model evaluation.

Use chronological training, validation, and test periods.

Evaluate using appropriate metrics, such as:

- MAE.
- RMSE.
- WAPE.
- R² where meaningful.

Avoid relying on MAPE where actual values can be zero or very close to zero.

Display model metrics based on held-out historical data, not training performance.

### Insufficient history

Do not generate misleading forecasts when historical data is insufficient.

Suggested initial quality gates:

- Daily forecasting: require at least six weeks of reasonably complete daily history.
- Monthly forecasting: require at least six complete monthly observations.

These gates should be configurable and validated against actual model performance.

If there is insufficient history, display a clear explanation and request more data.

Do not manufacture missing historical observations.

### Forecast isolation

Do not blindly combine:

- Different providers with different currencies.
- Actual and amortized costs.
- Different billing accounts when the analysis requires separate reporting.
- Daily and monthly observations.
- Incompatible datasets.

Combined forecasting is permitted only after verifying that the selected datasets have compatible cost semantics and currency treatment.

## 13.2 Model Artifacts

Save model artifacts with metadata.

Each model version should contain:

- Model identifier.
- Model type.
- Provider or series scope.
- Currency.
- Cost basis.
- Training data range.
- Feature list.
- Feature schema version.
- Training timestamp.
- Evaluation metrics.
- Model artifact path.
- Preprocessing configuration.

Validate feature compatibility before loading a model for prediction.

A model trained on AWS cost data must not silently be used on incompatible Azure or GCP features.

## 13.3 Google Colab Integration

Google Colab may be used for experiments or model training.

Use:

`notebooks/colab_training.ipynb`

For integration:

1. The notebook loads properly prepared billing data.
2. It performs chronological validation.
3. It trains and evaluates candidate models.
4. It saves the model and metadata.
5. It exports compatible artifacts.
6. The main application loads the exported model.
7. The application validates that the model supports the selected data.

Recommended model location:

`artifacts/models/staging/`

After successful validation, promote the model to:

`artifacts/models/production/`

Do not make the main application depend on an open Colab session.

---

# 14. Anomaly Detection

Start with interpretable statistical detection.

Possible methods:

- Rolling average.
- Rolling median.
- Percentage change threshold.
- Robust deviation scoring.
- Isolation Forest as an optional experiment.

The anomaly engine should identify:

- Unexpected daily or monthly cost increases.
- Significant service-level changes.
- Sudden account-level spending changes.
- Unusual spending patterns.
- Possible billing data quality issues.

Anomaly detection must consider data completeness.

A partially imported billing period or delayed billing records may create false anomalies.

Always distinguish a detected statistical anomaly from a confirmed infrastructure problem.

---

# 15. Optimization Recommendation Engine

The first version must use explainable rules.

Examples:

| Condition | Suggested recommendation |
|---|---|
| Service costs increase repeatedly | Investigate usage and deployment changes |
| Spending exceeds configured budget | Review spending and projected month-end costs |
| Service has a sudden cost spike | Inspect the affected service and billing period |
| Repeated storage spending increase | Review storage growth, retention, and lifecycle policies |
| High cost concentration in one service | Review allocation and optimization options |
| Cloud-native recommendation is available | Import and display the provider's recommendation, where supported |

Each rule must specify:

- Required data fields.
- Detection logic.
- Supporting evidence.
- Recommendation text.
- Severity or priority.
- Known limitations.

### CPU and memory limitations

Billing exports do not reliably provide CPU utilization or memory utilization.

Do not make rightsizing decisions using billing cost alone.

For future utilization-based recommendations, integrate:

- AWS CloudWatch.
- Azure Monitor.
- Google Cloud Monitoring.

Keep billing ingestion and telemetry ingestion as separate data sources.

### No automatic infrastructure changes

The MVP must never automatically shut down, resize, delete, or otherwise modify a cloud resource.

All recommendations are advisory and require human review.

---

# 16. User Interface and User Flow

Design the primary user journey as follows:

```text
USER OPENS CLOUDCOST IQ
          |
          v
     CREATES WORKSPACE
          |
          v
    OPENS DATA IMPORT
          |
          v
    SELECTS PROVIDER
          |
          v
    UPLOADS BILLING FILE
          |
          v
    VALIDATES BILLING DATA
          |
          v
    CONFIRMS IMPORT
          |
          v
    VIEWS COST DASHBOARD
          |
          v
    ANALYZES SPENDING
          |
          +---------------------+
          |                     |
          v                     v
   ML FORECASTING        ANOMALY DETECTION
          |                     |
          +----------+----------+
                     |
                     v
             RECOMMENDATIONS
                     |
                     v
              USER REVIEWS
                     |
                     v
             TRACKS THE RESULT
```

### Navigation

Use these main sections:

1. Overview.
2. Import Billing Data.
3. Cost Analysis.
4. ML Predictions.
5. Anomaly Detection.
6. Cost Optimization.
7. Import History.
8. Data Quality.
9. Settings.

### Interface behavior

- Show meaningful empty states when no data has been imported.
- Clearly label sample and demo data.
- Display loading and processing status.
- Show understandable error messages.
- Preserve selected filters when navigating where practical.
- Make charts interactive.
- Provide export options for useful tables.
- Never display sample results as real imported billing data.

---

# 17. API Requirements

Implement versioned REST endpoints.

Suggested endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/imports` | Upload billing files |
| GET | `/api/v1/imports` | List import batches |
| GET | `/api/v1/imports/{id}` | Get import details |
| GET | `/api/v1/costs/summary` | Get cost summary |
| GET | `/api/v1/costs/timeseries` | Get cost time series |
| GET | `/api/v1/costs/breakdown` | Get cost breakdown |
| POST | `/api/v1/forecasts` | Create a forecast |
| GET | `/api/v1/forecasts/{id}` | Retrieve forecast results |
| GET | `/api/v1/anomalies` | Retrieve detected anomalies |
| GET | `/api/v1/recommendations` | Retrieve recommendations |
| PATCH | `/api/v1/recommendations/{id}` | Update recommendation status |
| GET | `/api/v1/health` | Application health check |

Validate inputs and return structured error responses.

Do not expose internal server paths, credentials, raw exceptions, or sensitive billing records in error messages.

---

# 18. Security Requirements

Security is important because billing exports may contain account identifiers, resource names, tags, and commercially sensitive information.

Implement:

- Safe file validation.
- File size limits.
- Secure file handling.
- Upload isolation.
- Path traversal protection.
- Safe archive extraction.
- Input validation.
- Restricted application access to stored files.
- Environment-based configuration.
- Appropriate error handling.
- Safe logging practices.

Never:

- Hardcode cloud credentials.
- Commit `.env` files.
- Commit real billing data.
- Ask users to upload permanent cloud access keys.
- Store secrets in billing records.
- Execute uploaded files.
- expose raw billing files through public URLs.

For future direct cloud connections, prefer read-only permissions and short-lived or federated credentials where supported.

The local MVP must not require AWS, Azure, or GCP credentials to analyze uploaded billing files.

---

# 19. Database and Storage

Use:

**DuckDB:** analytical queries over processed billing data.  
**Parquet:** processed billing record storage.  
**SQLite:** application metadata and import tracking.

Suggested metadata tables:

### `import_batches`

Stores upload and processing history.

### `uploaded_files`

Stores file information and source metadata.

### `workspaces`

Stores workspace configuration.

### `model_versions`

Stores model metadata and evaluation results.

### `forecasts`

Stores generated forecast information.

### `anomalies`

Stores detected anomalies.

### `recommendations`

Stores recommendation details and status.

### `application_settings`

Stores safe application configuration.

Do not store raw cloud access credentials in SQLite.

Design storage access through repository or service modules rather than accessing database files directly from dashboard pages.

---

# 20. DevOps and MLOps

## 20.1 Docker

Provide a Dockerfile and Docker Compose configuration.

The local setup should support:

- Streamlit frontend.
- FastAPI backend.
- Persistent application data.
- Persistent model artifacts.
- Optional MLflow tracking service.

A developer should be able to start the application using a documented command.

Do not introduce Kubernetes for the initial deployment.

## 20.2 CI/CD

Use GitHub Actions to run:

1. Dependency and configuration checks.
2. Code linting.
3. Unit tests.
4. Ingestion tests.
5. ML tests.
6. Integration tests.
7. Docker build validation.

The pipeline must not need access to real customer billing data.

## 20.3 MLOps

Use MLflow for experiment tracking when the training pipeline is ready.

Track:

- Dataset version or fingerprint.
- Training period.
- Model parameters.
- Evaluation metrics.
- Feature definitions.
- Artifact location.

Do not upload sensitive billing data into publicly accessible experiment tracking.

Promote a model to production only after validating its compatibility and performance.

## 20.4 Application Monitoring

For later deployment, monitor:

- API availability.
- Request failures.
- File import failures.
- Processing time.
- Number of rejected records.
- Forecast generation failures.
- Model loading failures.
- Application memory usage.

Do not log sensitive billing details or account identifiers as metric labels.

---

# 21. Testing Requirements

Every major module must have automated tests.

## Ingestion tests

Test:

- Valid AWS files.
- Valid Azure files.
- Valid GCP files.
- Valid FOCUS files.
- CSV parsing.
- Parquet parsing.
- GZIP handling.
- Unsupported file formats.
- Empty files.
- Missing columns.
- Invalid cost values.
- Invalid dates.
- Multiple accounts.
- Multiple currencies.
- Large-file behavior.
- Repeated imports.

## Normalization tests

Test:

- Provider field mapping.
- Missing optional fields.
- Negative credits.
- Zero cost records.
- Actual versus amortized cost separation.
- Resource IDs.
- Tags.
- Invoice months.
- Usage dates.
- Currency preservation.

## Analytics tests

Test:

- Cost summaries.
- Date filtering.
- Provider filtering.
- Service breakdown.
- Account breakdown.
- Multiple currencies.
- Incomplete billing periods.
- Empty dataset behavior.

## ML tests

Test:

- Chronological data splitting.
- Baseline comparisons.
- Feature generation.
- Missing values.
- Insufficient history.
- Model serialization.
- Model loading.
- Prediction schema compatibility.
- Metric calculation.
- No future-data leakage.

## Security tests

Test:

- Invalid file uploads.
- Oversized file rejection.
- Unsafe ZIP entries.
- Path traversal attempts.
- Invalid API requests.
- Sensitive error-message exposure.

Use isolated and explicitly labelled fixtures for test billing data.

---

# 22. Development Phases

Develop the application in the following order.

## Phase 1 — Project Foundation

Create:

- Python environment.
- Folder structure.
- Application configuration.
- Streamlit application.
- FastAPI application.
- SQLite connection.
- DuckDB connection.
- Initial documentation.

Deliverable: application starts successfully with a basic navigation structure.

## Phase 2 — Billing Ingestion

Implement:

- AWS parser.
- Azure parser.
- GCP parser.
- FOCUS parser.
- File detection.
- Upload interface.
- Import metadata.
- File validation.

Deliverable: supported real billing exports can be uploaded and inspected.

## Phase 3 — Normalization and Data Quality

Implement:

- Common billing schema.
- Provider mappings.
- Currency validation.
- Cost-basis handling.
- Duplicate detection.
- Reimport handling.
- Rejected record reporting.
- Data lineage.

Deliverable: uploaded billing data can be safely normalized and audited.

## Phase 4 — Analytics

Implement:

- Cost summary.
- Time-series aggregation.
- Provider comparison.
- Service breakdown.
- Account and project breakdown.
- Cost filtering.

Deliverable: real billing data can be analyzed consistently.

## Phase 5 — Dashboard

Implement:

- Overview.
- Cost analysis.
- Import history.
- Data quality.
- Interactive charts.
- Empty and error states.

Deliverable: users can understand and investigate multi-cloud spending.

## Phase 6 — Machine Learning

Implement:

- Baseline forecasting.
- RandomForest forecasting.
- Chronological evaluation.
- Prediction generation.
- Model metadata.
- Model artifacts.
- Forecast UI.

Deliverable: users can view predictions with meaningful historical evaluation metrics.

## Phase 7 — Anomaly Detection and Recommendations

Implement:

- Statistical anomaly detection.
- Anomaly explanation.
- Rule-based recommendations.
- Recommendation tracking.

Deliverable: users can identify unusual spending and review potential optimization opportunities.

## Phase 8 — DevOps and MLOps

Implement:

- Docker.
- Docker Compose.
- GitHub Actions.
- MLflow integration.
- Automated testing.
- Application health monitoring.

Deliverable: reproducible development and deployment workflow.

## Phase 9 — Future Enhancements

Only after the first version is stable, consider:

- Direct AWS connections.
- Direct Azure connections.
- Direct GCP BigQuery integration.
- Automated billing refresh.
- Cloud telemetry integration.
- Team-based cost allocation.
- Cloud-native optimization recommendations.
- More sophisticated forecasting.
- Multi-user workspaces.

---

# 23. README Requirements

Maintain a README containing:

1. Product overview.
2. Main features.
3. Architecture.
4. Technology stack.
5. Supported billing formats.
6. Installation instructions.
7. Configuration instructions.
8. How to start the application.
9. How to upload AWS billing data.
10. How to upload Azure billing data.
11. How to upload GCP billing data.
12. How forecasting works.
13. How anomaly detection works.
14. Known limitations.
15. Testing instructions.
16. Future enhancements.

Document any unsupported provider export format instead of claiming universal compatibility.

---

# 24. Definition of Done

The project is not considered complete merely because the dashboard loads.

The MVP is complete when all the following conditions are met:

### Billing ingestion

- Users can upload supported AWS billing exports.
- Users can upload supported Azure billing exports.
- Users can upload supported GCP billing exports.
- The application detects and validates input files.
- Invalid data produces understandable warnings.
- Imports are tracked.
- Reimports do not silently double-count costs.

### Data processing

- Provider records are normalized into a common structure.
- Original source information is preserved.
- Actual and amortized costs remain separate.
- Different currencies are not incorrectly summed.
- Missing data is not silently converted into zero cost.

### Analytics

- The dashboard displays imported real billing information.
- Provider comparisons work.
- Service and account breakdowns work.
- Date filters work.
- Empty states and data limitations are visible.

### Machine learning

- Forecasts use historical billing data.
- Training and testing are chronological.
- Baseline metrics are available.
- Predictions are checked against compatible feature schemas.
- Insufficient history is handled safely.
- Metrics are calculated from held-out data.

### Recommendations

- Recommendations include evidence.
- Recommendations identify their limitations.
- Potential savings are not fabricated.
- Infrastructure is never changed automatically.

### Engineering quality

- Core functionality has automated tests.
- Application configuration is documented.
- Sensitive files are excluded from Git.
- Docker setup works.
- README instructions are accurate.
- No unnecessary Rust or microservice infrastructure has been introduced.

---

# 25. Final Product Principle

CloudCost IQ must be built around one central principle:

**Accurate billing data first, reliable analytics second, machine learning third, optimization recommendations fourth.**

A prediction based on incorrect billing data is not useful.

A recommendation without supporting evidence is not trustworthy.

A combined cloud spending figure that mixes currencies or cost bases is misleading.

A visually impressive dashboard is not a substitute for a reliable data pipeline.

Build CloudCost IQ as a practical, explainable, and extensible multi-cloud cost intelligence application.

Prioritize correctness, transparency, and a working end-to-end user journey over feature quantity.
