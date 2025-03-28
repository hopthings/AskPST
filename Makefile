.PHONY: setup test lint clean all

all: setup test

setup:
	pip install -e .
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	flake8 askpst tests
	black --check askpst tests
	isort --check askpst tests
	mypy askpst

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete