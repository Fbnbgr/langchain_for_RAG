#!/bin/sh
set -e

GND_FILE="/gnd_data/authorities_lds_20260217.nt"
TDB_DIR="/fuseki/databases/gnd"

# Prüfen ob wirklich Daten drin sind, nicht nur ob Ordner existiert
if [ -f "$TDB_DIR/Data-0001.ivm" ]; then
  echo "TDB2 already loaded, skipping."
  exit 0
fi

echo "Loading GND..."
/jena/bin/tdb2.tdbloader \
  --loc "$TDB_DIR" \
  "$GND_FILE"

echo "Done."