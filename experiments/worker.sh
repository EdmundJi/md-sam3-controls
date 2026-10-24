#!/bin/bash
# worker.sh QUEUE GPU -- pop jobs from a shared queue until it is empty AND the
# queue is closed (file QUEUE.open removed). Jobs whose --prune-mask-file does
# not exist yet are put back at the end of the queue.
Q=$(readlink -f "$1"); export GPU=${2:-0}
cd "$(dirname "$0")"
pop() { flock "$Q.lock" bash -c 'l=$(head -n1 "$0"); [ -n "$l" ] && sed -i 1d "$0"; printf "%s" "$l"' "$Q"; }
push() { flock "$Q.lock" bash -c 'printf "%s\n" "$1" >> "$0"' "$Q" "$1"; }
while true; do
  line=$(pop)
  if [ -z "$line" ]; then
    [ -e "$Q.open" ] && { sleep 30; continue; } || break
  fi
  mask=$(echo "$line" | sed -nE 's/.*--prune-mask-file ([^ ]+).*/\1/p')
  if [ -n "$mask" ] && [ ! -f "$mask" ]; then push "$line"; sleep 15; continue; fi
  set -- $line
  name=$1; script=$2; shift 2
  bash ./job.sh "$name" "$script" "$@"
done
echo "[$(date '+%F %T')] worker exit"
