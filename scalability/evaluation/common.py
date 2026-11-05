"""Evaluate community-control strategies using saved scenario inputs.

Load frozen PPO policy-normalizer pairs, run the selected controller,
and record physical trajectories, control metrics, and timing artifacts.
"""
import os,sys,json,time,pickle,csv,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from paths import ROOT, SCALE, inside, read_week
from dataclasses import replace
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE/'code'))
from pipeline import ec5,sim_config,env_config,project_allocation,corrected_mpc,PPO,torch,scipy

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def heartbeat(**kw):
 path=os.environ.get('E12_HEARTBEAT')
 if path:atomic_json(path,dict(pid=os.getpid(),timestamp=time.time(),**kw))
def times(values,prefix):
 x=np.asarray(values,float)
 return {prefix+k:float(v) for k,v in dict(total_seconds=x.sum(),mean_seconds=x.mean() if x.size else 0,median_seconds=np.median(x) if x.size else 0,std_seconds=x.std() if x.size else 0,p95_seconds=np.percentile(x,95) if x.size else 0,max_seconds=x.max() if x.size else 0).items()}
def valid_week(path):
 return read_week(path,require_solver=True)
class FrozenPolicy:
 def __init__(self,n,seed):
  self.n=n;self.seed=seed
  entry=next(x for x in json.loads((BASE/'configs/models.json').read_text())['models'] if x['N']==n and x['seed']==seed)
  entry=dict(entry,model=inside(entry['model']),vecnormalize=inside(entry['vecnormalize']))
  self.entry=entry
  assert digest(entry['model'])==entry['model_sha256'] and digest(entry['vecnormalize'])==entry['vecnormalize_sha256']
  self.model=PPO.load(entry['model'],device='cpu');self.model.policy.set_training_mode(False)
  with open(entry['vecnormalize'],'rb') as f:self.norm=pickle.load(f)
  self.norm.training=False;self.norm.norm_reward=False
  assert self.model.ent_coef==.003 and self.model.policy.net_arch=={'pi':[128,64],'vf':[128,64]}
  assert self.model.num_timesteps==5001216 and self.model.seed==seed
  assert self.model.observation_space.shape==(16*n+21,) and self.model.action_space.shape==(3*n,)
  self.original=(self.norm.obs_rms.mean.copy(),self.norm.obs_rms.var.copy(),self.norm.obs_rms.count)
  self.std=self.model.policy.log_std.detach().exp().cpu().numpy()
  for _ in range(5):self.model.predict(np.zeros(16*n+21,np.float32),deterministic=True)
 def strategy(self,ratios):
  self.seconds=[];self.normalized_seconds=[];self.raw=[];self.observations=[]
  def call(**kw):
   start=time.perf_counter_ns();obs=self.norm.normalize_obs(kw['observation']);t=time.perf_counter_ns()
   a,_=self.model.predict(obs,deterministic=True);end=time.perf_counter_ns()
   self.seconds.append((end-t)*1e-9);self.normalized_seconds.append((end-start)*1e-9)
   self.observations.append(obs.copy())
   n=self.n;out=dict(betas=project_allocation(ratios+np.clip(a[:n],-1,1)*.6).astype(np.float32),discharge_rates=np.clip((a[n:2*n]+1)/2,0,1),soc_reserves=np.clip((a[2*n:]+1)/2,0,1))
   return out
  return call
 def check_frozen(self):
  np.testing.assert_array_equal(self.original[0],self.norm.obs_rms.mean);np.testing.assert_array_equal(self.original[1],self.norm.obs_rms.var);assert self.original[2]==self.norm.obs_rms.count
  assert digest(self.entry['model'])==self.entry['model_sha256'] and digest(self.entry['vecnormalize'])==self.entry['vecnormalize_sha256']

def run_week(campaign,n,w,method,policy=None,max_decisions=168):
    if not 1 <= max_decisions <= 168: raise ValueError('Decisions must be 1..168')
    smoke=max_decisions<168
    root=inside(campaign)/f'N{n}';directory=root/method;directory.mkdir(parents=True,exist_ok=True)
    inputs=BASE/'inputs'/f'N{n}'
    done=directory/f'week_{w:02d}.json'
    if done.exists():
        recorded=json.loads(done.read_text())
        current=inputs/f'week_{w:02d}.npz'
        assert digest(current)==recorded['input_sha256'], 'Completed week input mismatch'
        print('RESUME: verified completed',done,flush=True)
        return valid_week(done)
    inp=inputs/f'week_{w:02d}.npz';info=json.loads(inp.with_suffix('.json').read_text());assert digest(inp)==info['sha256']
    with np.load(inp) as d: data={k:d[k] for k in d.files}
    cfg=sim_config(n); ec5.Device.delta_t(.25)
    profiles=[dict(load=data['load_w'][:,i],csv_name=info['csv_names'][i]) for i in range(n)]
    ratios=data['reference_ratios'];solves=[];decisions=[]
    attempt=directory/f'week_{w:02d}_attempt_{time.time_ns()}';attempt.mkdir()
    solverfile=attempt/'solver.jsonl'
    if method=='mpc':
        original=scipy.optimize.minimize
        def record(*args,**kw):
            size=np.asarray(args[1]).size
            assert size==24*n and len(kw['bounds'])==size and len(kw['constraints'])==8
            start=time.perf_counter_ns();sol=original(*args,**kw);elapsed=(time.perf_counter_ns()-start)*1e-9
            x=np.asarray(sol.x);beta=x[:8*n].reshape(8,n)
            record=dict(decision_index=len(decisions),start_index=len(solves)%3,seconds=elapsed,
                success=bool(sol.success),status=int(sol.status),message=str(sol.message),nit=int(sol.nit),
                iteration_limit=bool(sol.status==9),nfev=int(sol.nfev),objective=float(sol.fun) if np.isfinite(sol.fun) else None,
                variables=size,allocation_variables=8*n,dr_variables=8*n,sr_variables=8*n,
                finite=bool(np.isfinite(x).all() and np.isfinite(sol.fun)),
                raw_max_equality_error=float(np.abs(beta.sum(axis=1)-1).max()),
                raw_beta_min=float(beta.min()),raw_hint_min=float(x[8*n:].min()),raw_hint_max=float(x[8*n:].max()))
            solves.append(record)
            heartbeat(N=n,week=w,method=method,decision=len(decisions),solver_calls=len(solves))
            with solverfile.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
            return sol
        with patch.object(scipy.optimize,'minimize',record):
            strategy=corrected_mpc(profiles,data['total_pv_w'],data['price_hourly'],cfg,ratios,
                horizonte_horas=8,peso_equidad=.1,mode='supervisory',maxiter=40)
    elif method.startswith('ppo_'):strategy=policy.strategy(ratios)
    elif method=='uniform':strategy=ec5.uniform_beta_strategy
    elif method=='rule_based':strategy=ec5.create_static_supervisory_strategy(ec5.create_rule_based_beta_strategy(n,ratios),discharge_rate=.2,soc_reserve=0)
    elif method=='proportional':strategy=ec5.create_static_supervisory_strategy(lambda **kw:ratios.copy(),discharge_rate=.2,soc_reserve=0)
    else:raise ValueError(method)
    if smoke:
        cfg=replace(cfg,sim_hours=float(max_decisions))
    invalid_applied=[]
    def measured(**kw):
        start=time.perf_counter_ns();out=strategy(**kw)
        if isinstance(out,dict):out['betas']=project_allocation(out['betas']).astype(np.float32)
        else:out=project_allocation(out).astype(np.float32)
        decisions.append((time.perf_counter_ns()-start)*1e-9)
        beta_out=out['betas'] if isinstance(out,dict) else out
        bad=int(not np.isfinite(beta_out).all() or np.any(np.asarray(beta_out)<.01-1e-7) or abs(np.sum(beta_out)-1)>1e-6)
        if isinstance(out,dict):
            for key in ('discharge_rates','soc_reserves'):
                v=np.asarray(out[key]);bad+=int(not np.isfinite(v).all() or np.any(v<0) or np.any(v>1))
        invalid_applied.append(bad)
        heartbeat(N=n,week=w,method=method,decision=len(decisions))
        if method=='mpc':
            b=strategy.allocation_sequences[0]
            assert b.shape==(8,n) and (b>=.01-1e-7).all() and np.allclose(b.sum(axis=1),1.,atol=1e-7)
        return out
    start=time.perf_counter()
    result=ec5.run_scenario(method,profiles,data['total_pv_w'],data['price_hourly'],ratios,cfg,measured,data['base_price'])
    wall=time.perf_counter()-start
    length=4*max_decisions
    def history(get):return np.array([get(s) for s in result.sites],float).T
    arrays=dict(step=np.arange(length),elapsed_hours=np.arange(length)*.25,
        price_eur_per_kwh=np.repeat(data['price_hourly'],4)[:length],total_pv_w=data['total_pv_w'][:length],
        load_w=history(lambda s:s.load.Load_history),soc_pct=history(lambda s:s.battery.SOC_history[1:]),
        beta=np.asarray(result.beta_history),dr=np.asarray(result.discharge_rate_history),sr=np.asarray(result.soc_reserve_history),
        grid_import_w=history(lambda s:s.Pg_history),curtailment_w=history(lambda s:s.Export_blocked_history),
        battery_power_w=history(lambda s:s.battery.Pbat_history))
    arrays['grid_import_total_w']=arrays['grid_import_w'].sum(axis=1)
    for k,a in arrays.items():assert a.shape[0]==length and np.isfinite(a).all(),(n,w,method,k,a.shape)
    for k in ('load_w','soc_pct','beta','dr','sr','grid_import_w','curtailment_w','battery_power_w'):assert arrays[k].shape==(length,n)
    assert np.allclose(arrays['load_w'],data['load_w'][:length]),'Input/physical load mismatch'
    b=arrays['beta'];dr=arrays['dr'];sr=arrays['sr'];soc=arrays['soc_pct']
    violations=int(np.sum(b<.01-1e-7)+np.sum(abs(b.sum(axis=1)-1)>1e-6)+np.sum(dr<0)+np.sum(dr>1)+np.sum(sr<0)+np.sum(sr>1)+np.sum(soc<10-1e-5)+np.sum(soc>90+1e-5))
    assert violations==0 and result.total_cost>=0 and np.isfinite(result.total_cost)
    throughput=float(abs(arrays['battery_power_w']).sum()*.25/1000)
    curtail=float(arrays['curtailment_w'].sum()*.25/1000)
    row=dict(N=n,week=w,seed=41+w,controller=method,cost_eur=float(result.total_cost),grid_import_kwh=float(result.grid_import_kwh),
        pv_curtailment_kwh=curtail,pv_curtailment_ratio=curtail/(float(arrays['total_pv_w'].sum())*.25/1000) if arrays['total_pv_w'].sum()>0 else None,
        jain=result.fairness_metrics['jain_index'],mean_abs_beta_bar_minus_r=result.fairness_metrics['deviation_from_pv_ratio'],
        battery_throughput_kwh=throughput,equivalent_full_cycles=throughput/(2*info['battery_energy_kwh']),
        simulation_wall_seconds=wall,decision_calls=len(decisions),beta_min=float(b.min()),beta_max=float(b.max()),
        beta_sum_max_error=float(abs(b.sum(axis=1)-1).max()),dr_min=float(dr.min()),dr_max=float(dr.max()),
        sr_min=float(sr.min()),sr_max=float(sr.max()),soc_min=float(soc.min()),soc_max=float(soc.max()),constraint_violations=violations,
        optimization_calls=len(solves),solver_failures=sum(not s['success'] for s in solves),solver_nonfinite=sum(not s['finite'] for s in solves),
        solver_iterations=sum(s['nit'] for s in solves),solver_function_evaluations=sum(s['nfev'] for s in solves),
        allocation_variables=8*n if method=='mpc' else 0,dr_variables=8*n if method=='mpc' else 0,sr_variables=8*n if method=='mpc' else 0,total_variables=24*n if method=='mpc' else 0,
        **times(decisions,'controller_'),**times([s['seconds'] for s in solves],'optimization_call_'))
    inference=policy.seconds if method.startswith('ppo_') else []
    row.update(policy_calls=len(inference),**times(inference,'inference_'))
    assert len(decisions)==length//4
    if method.startswith('ppo_'):
        policy.check_frozen();assert len(inference)==length//4
    if method=='mpc':
        assert len(solves)==3*len(decisions)
        row.update(times([sum(s['seconds'] for s in solves if s['decision_index']==i) for i in range(len(decisions))],'optimization_decision_'))
    high=arrays['price_eur_per_kwh']>=np.quantile(arrays['price_eur_per_kwh'],.75)
    row.update(soc_mean=float(soc.mean()),sr_mean=float(sr.mean()),dr_mean=float(dr.mean()),
        high_price_soc=float(soc[high].mean()),high_price_sr=float(sr[high].mean()),high_price_dr=float(dr[high].mean()),
        high_price_discharge_kwh=float(np.maximum(-arrays['battery_power_w'][high],0).sum()*.25/1000),
        beta_floor_pct=float(np.mean(b<=.010001)*100),applied_invalid_decisions=sum(invalid_applied),
        training_seed=policy.seed if policy else None,input_sha256=info['sha256'])
    if method.startswith('ppo_'):
        with torch.no_grad():policy.raw=policy.model.policy.get_distribution(torch.as_tensor(np.asarray(policy.observations),dtype=torch.float32)).distribution.mean.cpu().numpy()
        # Compute untimed reward diagnostics using the stored reference-cost series.
        ref=np.load(inputs/f'week_{w:02d}_reward_reference.npy')[:length]
        physical_cost=np.array([s.Cost_history for s in result.sites]).sum(axis=0)
        advantage=(ref-physical_cost).reshape(-1,4).mean(axis=1)
        row.update(economic_clip_pct=float(np.mean(abs(advantage)>5*n/3)*100),economic_clip_limit=5*n/3,
            raw_clip_pct=float(np.mean(abs(np.asarray(policy.raw))>1)*100),
            **{f'policy_std_{head}':float(policy.std[i*n:(i+1)*n].mean()) for i,head in enumerate(('beta','dr','sr'))},
            **times(policy.normalized_seconds,'normalized_predict_'))
        arrays.update(raw_actions=np.asarray(policy.raw),economic_advantage=advantage,normalized_predict_seconds=np.asarray(policy.normalized_seconds))
    arrays.update(decision_seconds=np.asarray(decisions),inference_seconds=np.asarray(inference))
    np.savez_compressed(attempt/'timeseries.npz',**arrays)
    artifacts={str(p.relative_to(directory)):digest(p) for p in attempt.iterdir() if p.is_file()}
    atomic_json(done,dict(complete=True,input_sha256=info['sha256'],metrics=row,artifacts=artifacts,
        time_convention='relative hours; SOC after transition; power averaged per step',smoke=smoke))
    print(f'COMPLETED N={n} week={w} controller={method}',flush=True)
    return row

