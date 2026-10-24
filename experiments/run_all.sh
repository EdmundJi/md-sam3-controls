#!/bin/bash
# run_all.sh [WORKERS] -- N workers on GPU 0 over queue.txt (seeded with EV17
# jobs). Append EV18 jobs later with: cat ev18_jobs.txt >> queue.txt ; then
# close the queue with: rm queue.txt.open
N=${1:-4}
cd "$(dirname "$0")"
bash gen_jobs.sh
cp ev17_jobs.txt queue.txt; touch queue.txt.open
for i in $(seq 1 $N); do bash worker.sh queue.txt 0 & sleep 15; done
wait
echo "[$(date '+%F %T')] ALL DONE"
