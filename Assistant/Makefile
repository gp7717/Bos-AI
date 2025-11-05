.PHONY: install run test format lint clean

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

format:
	black app/ tests/
	ruff check app/ tests/ --fix

lint:
	ruff check app/ tests/
	mypy app/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

