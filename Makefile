.PHONY: setup run clean distclean test demo api infra-up infra-down

setup:
	python3 -m venv rag_pipeline_env
	rag_pipeline_env/bin/pip install -r requirements.txt
	rag_pipeline_env/bin/pip install sentence-transformers

run:
	rag_pipeline_env/bin/python main.py --run

demo:
	rag_pipeline_env/bin/python src/demo/inspect_data.py

api:
	rag_pipeline_env/bin/uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8888

ui:
	rag_pipeline_env/bin/streamlit run src/ui/app.py
streaming-producer:
	PYTHONPATH=. rag_pipeline_env/bin/python src/streaming/producer.py

streaming-consumer:
	PYTHONPATH=. rag_pipeline_env/bin/python src/streaming/consumer.py --hyde

infra-up:
	docker-compose up -d

infra-down:
	docker-compose down
clean:
	rm -rf output
	find . -type d -name "__pycache__" -exec rm -rf {} +

distclean: clean
	rm -rf rag_pipeline_env

test:
	rag_pipeline_env/bin/pytest tests/
