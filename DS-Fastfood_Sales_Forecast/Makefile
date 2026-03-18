.PHONY: setup data train test all clean

all: setup data train test

setup:
	pip install -r requirements.txt
	pip install -e .

data:
	python src/etl.py
	python src/features.py

train:
	python src/train.py

test:
	pytest -v tests/

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf logs/*
