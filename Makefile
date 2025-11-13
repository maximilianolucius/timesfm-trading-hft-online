PYTHON ?= python
VENV ?= .venv

.PHONY: venv install lint format typecheck test prep train backtest serve trade

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .[quant]

lint:
	. $(VENV)/bin/activate && ruff check .

format:
	. $(VENV)/bin/activate && ruff format .

typecheck:
	. $(VENV)/bin/activate && mypy .

test:
	. $(VENV)/bin/activate && pytest -q

prep:
	bin/tft prep --config $(CONFIG)

train:
	bin/tft train --config $(CONFIG)

backtest:
	bin/tft backtest --config $(CONFIG)

serve:
	bin/tft serve --config $(CONFIG)

trade:
	bin/tft trade --config $(CONFIG)
