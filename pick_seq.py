"""Pick candidate instances for the click-sequence figure (Fig. 5).

Wanted: a camouflage instance where DGC converges within a few clicks while
farthest-point clearly does not, so a 4-click strip shows a visible gap.
Reads the existing v2 CSVs only -- no GPU, no new inference.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import json
import os

V2 = rf"{ROOT}/results/noc_v2"
SETS = ["camo", "chameleon"]
BB = "vit_b"
CAP = 21


def load(strat, ds):
    p = f"{V2}/noc_{BB}_{strat}_{ds}.csv"
    if not os.path.exists(p):
        print("MISSING", p)
        return {}
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            traj = json.loads(r.get("iou_traj", "[]"))
        except Exception:
            traj = []
        out[r["id"]] = (min(int(r["n_clicks_90"]), CAP), traj)
    return out


first = True
rows = []
for ds in SETS:
    risk, fps = load("risk_dt", ds), load("fps", ds)
    if first and risk:
        p = f"{V2}/noc_{BB}_risk_dt_{ds}.csv"
        print("CSV columns:", next(csv.reader(open(p, encoding="utf-8"))))
        first = False
    for k in sorted(set(risk) & set(fps)):
        rn, rt = risk[k]
        fn, ft = fps[k]
        if len(rt) < 5 or len(ft) < 5:
            continue
        # DGC should converge fast; fps should still be behind at click 4
        if not (2 <= rn <= 6):
            continue
        gap4 = rt[3] - ft[3]          # IoU advantage at the 4th click
        rows.append(dict(ds=ds, id=k, risk_noc=rn, fps_noc=fn,
                         iou1=round(rt[0], 3),
                         risk4=round(rt[3], 3), fps4=round(ft[3], 3),
                         gap4=round(gap4, 3), noc_gap=fn - rn))

rows.sort(key=lambda r: (-r["gap4"], r["risk_noc"]))
print(f"\n{len(rows)} candidates (DGC NoC@90 in 2..6, both trajectories >=5 long)")
print(f"{'dataset':11s} {'id':28s} {'IoU@c1':>7s} {'DGC@c4':>7s} {'FPS@c4':>7s} "
      f"{'gap@c4':>7s} {'NoC r/f':>9s}")
for r in rows[:25]:
    print(f"{r['ds']:11s} {r['id'][:28]:28s} {r['iou1']:7.3f} {r['risk4']:7.3f} "
          f"{r['fps4']:7.3f} {r['gap4']:7.3f} {r['risk_noc']:4d}/{r['fps_noc']:<4d}")
print("PICK_DONE")
