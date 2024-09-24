#!/bin/bash
set -euo pipefail

# prereqs: 
# apt install libpq-dev
# pip install pyscopg
# apt install postgresql 
db_name="$1"
createdb "$db_name"
psql "$db_name" -c "CREATE EXTENSION IF NOT EXISTS vector;"