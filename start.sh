#!/bin/bash
uvicorn web_api:app --host 127.0.0.1  --port 6288 --reload --workers 1