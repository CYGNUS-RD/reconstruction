#!/bin/bash

# Controllo argomenti
if [ $# -lt 2 ]; then
    echo "Usage: $0 CEID [-f|-r]"
    exit 1
fi

CEID=$1
ACTION=$2   # -f o -r

# Verifica che l'azione sia valida
if [[ "$ACTION" != "-f" && "$ACTION" != "-r" ]]; then
    echo "Invalid action: $ACTION. Use -f or -r"
    exit 1
fi

# Lista job IDs in un array
mapfile -t JOBS < <(cygno_htc -q "$CEID" | grep "ID:" | awk '{print $3}')
TOTAL=${#JOBS[@]}

if [ $TOTAL -eq 0 ]; then
    echo "No jobs found for CE $CEID"
    exit 0
fi

# Loop sui job con contatore
COUNT=0
for jid in "${JOBS[@]}"; do
    COUNT=$((COUNT+1))
    echo "[$COUNT/$TOTAL] Executing $ACTION on job $jid CE $CEID"
    cygno_htc "$ACTION" "$jid" "$CEID"
done
