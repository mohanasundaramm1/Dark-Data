.PHONY: setup run clean test demo

setup:
	python3 -m venv rag_pipeline_env
	rag_pipeline_env/bin/pip install -r requirements.txt

run:
	rag_pipeline_env/bin/python main.py --run

demo:
	rag_pipeline_env/bin/python src/demo/inspect_data.py

clean:
	rm -rf output
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf rag_pipeline_env

test:
	rag_pipeline_env/bin/pytest tests/
