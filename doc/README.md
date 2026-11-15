# CloudCost IQ Documentation

This folder contains project documentation (mirrored from `docs/`).

## Index
- [Architecture](architecture.md) — system components, data flow, storage layout
- [User Guide](user_guide.md) — UI pages, workflows, manual demo datasets, troubleshooting
- [Ingestion Guide](ingestion_guide.md) — supported providers/formats (AWS/Azure/GCP/FOCUS 1.x), validation, safety, cost semantics
- [Data Dictionary](data_dictionary.md) — normalized billing schema and field semantics
- [Dataset Sources](dataset_sources.md) — ingestion/dashboard/training eligibility and source rules
- [ML Training Guide](ml_training_guide.md) — data prep, features, training, evaluation, Colab, MLflow, promotion
- [ML Pipeline](ml_pipeline.md) — end-to-end stages from normalized data to validated inference
- [Model Card](model_card.md) — required artifact metadata and example candidate notes

See also the root [`README.md`](../README.md), [`Makefile`](../Makefile), and [`AGENTS.md`](../AGENTS.md).
