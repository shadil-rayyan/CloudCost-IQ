.PHONY: help install install-dev install-ml run-ui run-api test demo-test lint clean

PYTHON ?= python
HOST ?= 127.0.0.1
API_PORT ?= 8000
UI_PORT ?= 8501

# Ensure project root is importable for `app.*` imports.
export PYTHONPATH := $(CURDIR)

help:
	@echo "CloudCost IQ - common commands"
	@echo "  make install       Install app dependencies"
	@echo "  make install-dev   Install app + dev/test dependencies"
	@echo "  make install-ml    Install ML training dependencies"
	@echo "  make run-ui        Start Streamlit dashboard (http://localhost:$(UI_PORT))"
	@echo "  make run-api       Start FastAPI server (http://$(HOST):$(API_PORT), health: /api/v1/health)"
	@echo "  make test          Run test suite"
	@echo "  make demo-test     Run manual-dataset UI/feature demo test (verbose)"
	@echo "  make lint          Run ruff checks"
	@echo "  make clean         Remove Python/cache artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

install-ml:
	$(PYTHON) -m pip install -r requirements-ml.txt

run-ui:
	$(PYTHON) -m streamlit run app/main.py --server.port $(UI_PORT)

run-api:
	$(PYTHON) -m uvicorn app.api.main:app --host $(HOST) --port $(API_PORT) --reload

test:
	$(PYTHON) -m pytest -q

demo-test:
	$(PYTHON) -m pytest tests/ui/test_manual_feature_demo.py -v -s

lint:
	$(PYTHON) -m ruff check app scripts tests

clean:
	$(PYTHON) -c "import pathlib,shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]" || true
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc') if p.is_file()]" || true
	$(PYTHON) -c "import shutil,pathlib; p=pathlib.Path('.pytest_cache'); shutil.rmtree(p, ignore_errors=True)" || true
