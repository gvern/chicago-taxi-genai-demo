.PHONY: test lint install-dev

test:
	python3 -m pytest --maxfail=1 --disable-warnings -q test/

lint:
	ruff check src/ test/

install-dev:
	pip install -r requirements.txt
	pip install pytest ruff
