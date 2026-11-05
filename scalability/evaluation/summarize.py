"""Aggregate complete selected results; no simulator or policy imports."""
import argparse,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from paths import SCALE,inside,new_output,read_week

def aggregate(source,B,rebuild_timing=False):
 import numpy as np
 import pandas as pd
 methods=['uniform','proportional','rule_based','ppo_42','ppo_142','ppo_242','mpc'];rows=[];solver=[];decision_times=[]
 for n in [3,5,10]:
  for method in methods:
   for week in range(1,11):
    p=source/f'N{n}/{method}/week_{week:02d}.json';d=json.loads(p.read_text());r=read_week(p,require_solver=rebuild_timing)
    info=json.loads((SCALE/f'inputs/N{n}/week_{week:02d}.json').read_text());assert d['input_sha256']==info['sha256']
    r=dict(r,method='PPO E12' if method.startswith('ppo_') else method)
    rows.append(r)
    for artifact in d['artifacts']:
     q=inside(p.parent/artifact)
     if q.name=='solver.jsonl' and rebuild_timing:
      solver.extend(dict(N=n,week=week,**json.loads(l)) for l in q.read_text().splitlines())
     if q.name=='timeseries.npz' and rebuild_timing:
      with np.load(q) as z:
       for key,label in [('inference_seconds','policy.predict'),('normalized_predict_seconds','normalization + predict'),('decision_seconds','controller')]:
        if key in z:
         decision_times.extend(dict(N=n,method=r['method'],training_seed=r['training_seed'],week=week,scope=label,seconds=float(t)) for t in z[key])
 assert len(rows)==210
 df=pd.DataFrame(rows);df.to_csv(B/'costs_by_scenario.csv',index=False)
 if rebuild_timing:
  sol=pd.DataFrame(solver);sol.to_csv(B/'mpc_solver_diagnostics.csv',index=False)
 paired=[]
 for n in [3,5,10]:
  for seed in [42,142,242]:
   for week in range(1,11):
    p=df[(df.N==n)&(df.training_seed==seed)&(df.week==week)].iloc[0]
    for ref in ['uniform','proportional','rule_based','mpc']:
     b=df[(df.N==n)&(df.method==ref)&(df.week==week)].iloc[0];assert p.input_sha256==b.input_sha256
     paired.append(dict(N=n,training_seed=seed,week=week,reference=ref,ppo_cost=p.cost_eur,reference_cost=b.cost_eur,difference_eur=p.cost_eur-b.cost_eur,difference_pct=100*(p.cost_eur-b.cost_eur)/b.cost_eur))
 paired=pd.DataFrame(paired);paired.to_csv(B/'paired_comparisons.csv',index=False)
 # Ratios are calculated per paired scenario first; no pooling of the 30 PPO episodes as training replicates.
 uniform=df[df.method=='uniform'][['N','week','cost_eur']].rename(columns={'cost_eur':'uniform_cost'})
 df=df.merge(uniform,on=['N','week'],validate='many_to_one');df['reduction_vs_uniform_pct']=100*(df.uniform_cost-df.cost_eur)/df.uniform_cost
 df.to_csv(B/'costs_by_scenario.csv',index=False)
 numeric=[x for x in df.select_dtypes('number').columns if x not in ['N','week','seed','training_seed']]
 ppo=df[df.method=='PPO E12'].groupby(['N','training_seed'])[numeric].mean().reset_index();ppo.to_csv(B/'ppo_by_seed.csv',index=False)
 summary=[]
 for n in [3,5,10]:
  for method in ['uniform','proportional','rule_based','PPO E12','mpc']:
   scenario=df[(df.N==n)&(df.method==method)];rep=ppo[ppo.N==n] if method=='PPO E12' else scenario
   row=dict(N=n,method=method,training_seeds=3 if method=='PPO E12' else 0,scenarios_per_seed=10)
   for k in numeric:
    row[k+'_mean']=float(rep[k].mean());row[k+'_sd_between_seeds']=float(rep[k].std(ddof=1)) if method=='PPO E12' else np.nan
   row['cost_sd_across_scenarios']=float(scenario.cost_eur.std(ddof=1)) if method!='PPO E12' else np.nan
   row['constraint_violations_total']=int(scenario.constraint_violations.sum());row['applied_invalid_decisions_total']=int(scenario.applied_invalid_decisions.sum())
   summary.append(row)
 summary=pd.DataFrame(summary);summary.to_csv(B/'summary_by_N_method.csv',index=False)
 df[['N','method','training_seed','week','jain','battery_throughput_kwh','equivalent_full_cycles','soc_mean','dr_mean','sr_mean','high_price_soc','high_price_sr','high_price_dr','high_price_discharge_kwh','pv_curtailment_kwh','grid_import_kwh','constraint_violations','applied_invalid_decisions']].to_csv(B/'fairness_cycling.csv',index=False)
 if rebuild_timing:
  times=pd.DataFrame(decision_times)
  # Solver calls and per-decision sum of the three starts are separate scopes.
  solver_times=sol.rename(columns={'seconds':'seconds'})[['N','week','decision_index','seconds']].copy()
  extra=[]
  for _,r in solver_times.iterrows():extra.append(dict(N=int(r.N),week=int(r.week),method='mpc',training_seed=np.nan,scope='solver call',seconds=r.seconds))
  for (n,w,d),g in solver_times.groupby(['N','week','decision_index']):extra.append(dict(N=n,week=w,method='mpc',training_seed=np.nan,scope='solver decision (three starts)',seconds=g.seconds.sum()))
  times=pd.concat([times,pd.DataFrame(extra)],ignore_index=True);times.to_csv(B/'timing_by_decision.csv',index=False)
  timing=times.groupby(['N','method','scope']).seconds.agg(count='count',mean='mean',median='median',p95=lambda x:x.quantile(.95),maximum='max').reset_index();timing.to_csv(B/'timing_scalability.csv',index=False)
  diag=sol.groupby('N').agg(calls=('success','size'),successes=('success','sum'),iteration_limits=('iteration_limit','sum'),finite=('finite','sum'));diag.to_csv(B/'mpc_solver_summary.csv')

 (B/"aggregation_status.json").write_text(json.dumps({
  "source":str(source.relative_to(SCALE.parent)),
  "episodes":210,
  "timing_rebuilt":rebuild_timing,
  "solver_documentary_integrity":"strict hashes checked" if rebuild_timing else "not validated; 30 reference solver hashes remain discrepant",
  "note":"Stored metric fields are retained. Detailed timing reconstruction requires solver records and strict artifact hashes."
 },indent=2),encoding="utf-8")

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--input",default="scalability/reference/episodes")
 parser.add_argument("--output",required=True)
 parser.add_argument("--rebuild-timing",action="store_true",help="Requires all solver artifact hashes to match. Currently blocked for saved reference solver records.")
 args=parser.parse_args()
 source=inside(args.input)
 # Require all 210 complete episodes; never combine truncated runs with reference results.
 for n in (3,5,10):
  for method in ("uniform","proportional","rule_based","ppo_42","ppo_142","ppo_242","mpc"):
   for week in range(1,11):read_week(source/f"N{n}/{method}/week_{week:02d}.json",require_solver=args.rebuild_timing)
 output=new_output(args.output)
 aggregate(source,output,args.rebuild_timing)

if __name__=="__main__":main()
