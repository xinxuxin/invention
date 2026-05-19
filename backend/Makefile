.PHONY: dev test lint

dev:
	uvicorn app.main:app --reload

test:
	pytest

lint:
	ruff check app tests
