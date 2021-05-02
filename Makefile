.PHONY: install-requirements
install-requirements: ## Install all requirements needed for development
	python -m pip install --upgrade pip~=20.2.0
	python -m pip install -r requirements.txt

.PHONY: item
item:
	python ./common/new_item.py -p $(path)

.PHONY: function
function:
	python ./common/item_to_function.py -i $(item) -o $(item)

.PHONY: test-py
test-py:
	python ./common/test_suite.py -r . -s py

.PHONY: test-ipynb
test-ipynb:
	python ./common/test_suite.py -r . -s ipynb

.PHONY: test-example
test-example:
	python ./common/test_suite.py -r . -s example