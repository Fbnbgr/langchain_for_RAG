#!/bin/sh
set -e

GND_FILE="/data/gnd/authorities-gnd.nt"
TDB_DIR="/fuseki/databases/gnd"

# Nur laden wenn TDB2-Verzeichnis noch leer ist (Idempotenz)
if [ -d "$TDB_DIR" ] && [ "$(ls -A $TDB_DIR)" ]; then
  echo "TDB2 database already exists, skipping load."
  exit 0
fi

echo "Starting GND bulk load (~9M Triples, dauert 10-30 min)..."
/jena/bin/tdb2.tdbloader \
  --loc "$TDB_DIR" \
  --graph "https://d-nb.info/gnd/" \
  "$GND_FILE"

echo "Load complete."