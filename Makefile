.PHONY: test coverage clean

test:
	python -m unittest discover

coverage:
	coverage run -m unittest discover
	coverage report
	coverage html

coverage-burst:
	coverage run -m unittest tests.test_burst_detector
	coverage report starguard/analyzers/burst_detector.py
	coverage html

clean:
	rm -rf .coverage htmlcov
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
