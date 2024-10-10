#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
user_name="$1"
db_name="$2"
out="$3"

erfile="$db_name.er"
#eralchemy -i "postgresql://$1@/$2" -o semantic_sign_language_search_db_schema.er
#eralchemy -i semantic_sign_language_search_db_schema.er -o semantic_sign_language_search_db_schema.pdf
eralchemy -i "postgresql://$user_name@/$db_name" -o "$erfile"
eralchemy -i "$erfile" -o "$out".pdf
