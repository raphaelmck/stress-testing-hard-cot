#!/usr/bin/env python3
"""R007 analysis: steering outcomes on `test` (D013).

Reads the frozen generations from r007_steer_test and reports, in the order
D013 froze them: the primary P(</think> within 60) contrast, the dose
response, the 1-19 / 20-60 breakdown (20-60 is the released YES window),
baseline validation against the released labels, label-stratified effects,
tokens to termination, and the matched-norm orthogonal controls.

Replicates are aggregated to a per-prefix rate first; bootstraps resample
question_id clusters, carrying all prefixes and replicates of a question
together, and condition deltas use the same resampled questions.
"""
import csv, json, sys, numpy as np, pathlib
sys.path.insert(0,'src'); import task1_data as T
rows=list(csv.DictReader(open('artifacts/runs/r007_steer_test/generations.csv')))
for r in rows:
    r['think_pos']=int(r['think_pos']); r['replicate']=int(r['replicate'])
    r['Y']=1.0 if r['think_pos']>0 else 0.0
    r['Y_1_19']=1.0 if 0<r['think_pos']<20 else 0.0
    r['Y_20_60']=1.0 if 20<=r['think_pos']<=60 else 0.0
conds=sorted({r['condition'] for r in rows}, key=lambda c:(c[0]!='b', float(c.split('a')[-1] if c.startswith('beta') else c[5:])))
order=['beta-2','beta-1','beta+0','beta+1','beta+2','ortho-2','ortho+2']
print(f"{len(rows)} rows, {len({r['filename'] for r in rows})} prefixes, conditions={sorted({r['condition'] for r in rows})}")

def prefix_rates(cond,key='Y',sel=None):
    d={}
    for r in rows:
        if r['condition']!=cond: continue
        if sel and not sel(r): continue
        d.setdefault(r['filename'],[]).append(r[key])
    return {k:float(np.mean(v)) for k,v in d.items()}
qof={r['filename']:r['question_id'] for r in rows}
lab={r['filename']:r['label'] for r in rows}

def boot(rate_by_cond, n=2000, seed=0):
    qs=sorted({qof[f] for f in next(iter(rate_by_cond.values()))})
    byq={}
    for f in next(iter(rate_by_cond.values())): byq.setdefault(qof[f],[]).append(f)
    rng=np.random.default_rng(seed); draws=[rng.choice(qs,len(qs),replace=True) for _ in range(n)]
    out={}
    for c,rt in rate_by_cond.items():
        v=np.array([np.mean([rt[f] for q in d for f in byq[q]]) for d in draws])
        out[c]=(float(np.mean(list(rt.values()))),*np.percentile(v,[2.5,97.5]))
    return out,draws,byq

print("\n=== primary: P(</think> within 60), macro over prefixes ===")
rt={c:prefix_rates(c) for c in order}
res,draws,byq=boot(rt)
for c in order:
    p,lo,hi=res[c]; print(f"  {c:8s} {p:.3f} [{lo:.3f}, {hi:.3f}]")
def pdelta(a,b,draws,byq):
    d=np.array([np.mean([rt[a][f] for q in dr for f in byq[q]])-np.mean([rt[b][f] for q in dr for f in byq[q]]) for dr in draws])
    return float(np.mean(list(rt[a].values()))-np.mean(list(rt[b].values()))), *np.percentile(d,[2.5,97.5]), float((d>0).mean())
for a,b in [('beta+2','beta-2'),('beta+2','beta+0'),('beta-2','beta+0'),('ortho+2','ortho-2'),('ortho+2','beta+0')]:
    pt,lo,hi,pp=pdelta(a,b,draws,byq); print(f"  delta {a} - {b}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] P(>0)={pp:.2f}")

for key,name in [('Y_1_19','tokens 1-19'),('Y_20_60','tokens 20-60 (released YES window)')]:
    print(f"\n=== {name} ===")
    rt2={c:prefix_rates(c,key) for c in order}
    r2,_,_=boot(rt2)
    print("  "+"  ".join(f"{c}={r2[c][0]:.3f}" for c in order))

print("\n=== baseline validation: unsteered vs released labels ===")
b=prefix_rates('beta+0')
yes=[v for f,v in b.items() if lab[f]=='yes']; no=[v for f,v in b.items() if lab[f]=='no']
print(f"  released YES prefixes (n={len(yes)}): P(term<=60) = {np.mean(yes):.3f}")
print(f"  released NO  prefixes (n={len(no)}): P(term<=60) = {np.mean(no):.3f}")
b2=prefix_rates('beta+0','Y_20_60')
print(f"  released YES, in 20-60 window: {np.mean([v for f,v in b2.items() if lab[f]=='yes']):.3f}  "
      f"NO: {np.mean([v for f,v in b2.items() if lab[f]=='no']):.3f}")

print("\n=== label-stratified steering ===")
for L in ('no','yes'):
    print(f"  released-{L.upper()} prefixes:")
    for c in order:
        rr=prefix_rates(c,sel=lambda r,L=L: r['label']==L)
        print(f"    {c:8s} {np.mean(list(rr.values())):.3f}")

print("\n=== tokens to termination (among terminated) ===")
for c in order:
    v=[r['think_pos'] for r in rows if r['condition']==c and r['think_pos']>0]
    print(f"  {c:8s} n={len(v):4d} median={np.median(v) if v else float('nan'):.1f} mean={np.mean(v) if v else float('nan'):.1f}")

summary={'primary':{c:res[c] for c in order},
         'delta_beta_plus2_minus_minus2':pdelta('beta+2','beta-2',draws,byq),
         'delta_ortho_plus2_minus_minus2':pdelta('ortho+2','ortho-2',draws,byq),
         'baseline_released_yes':float(np.mean(yes)),'baseline_released_no':float(np.mean(no))}
pathlib.Path('artifacts/runs/r007_steer_test/metrics.json').write_text(json.dumps(summary,indent=2))
print("\n=== identical-continuation check vs baseline (CRN) ===")
base={(r['filename'],r['replicate']):r['think_pos'] for r in rows if r['condition']=='beta+0'}
for c in order:
    same=sum(1 for r in rows if r['condition']==c and base[(r['filename'],r['replicate'])]==r['think_pos'])
    tot=sum(1 for r in rows if r['condition']==c)
    print(f"  {c:8s} same think_pos as baseline: {same}/{tot} = {same/tot:.1%}")
