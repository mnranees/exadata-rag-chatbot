# SR JSON RAG

Input: `sr_json/*.json`\nOutput: `vectordb_sr/`\nCollection: `service_requests`\n\nRun:\n```bash\npython sr_ingest.py --sr-json-dir ./sr_json --chroma-dir ./vectordb_sr --collection service_requests\n```\n\nDo not commit the SR JSON files or credentials to Git unless the repository is approved for this internal data.\n