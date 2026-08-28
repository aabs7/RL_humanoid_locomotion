import csv, math, shutil, torch
from pathlib import Path
from rl.logger import Logger

run = Path("runs/_smoke"); shutil.rmtree(run, ignore_errors=True); run.mkdir(parents=True)
lg = Logger(run, print_every=25)
for i in range(50):
    lg.log("losses/policy_loss", math.sin(i / 5))
    lg.log("losses/value_loss", math.cos(i / 5))
    if i % 10 == 0:                       # a metric that appears late and sporadically
        lg.log("eval/return", -50 + i)
    lg.histogram("policy/actions", torch.randn(64), i)
    lg.dump(step=i * 2048)
lg.close()

rows = list(csv.DictReader(open(run / "metrics.csv")))
assert len(rows) == 50
assert sum(1 for r in rows if r["eval/return"]) == 5      # sparse column, blank elsewhere
print(list(rows[0].keys()))
