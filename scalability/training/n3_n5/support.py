"""Published scalability PPO training support."""
import os,sys,json,time,csv,hashlib,pickle
from pathlib import Path
from dataclasses import asdict
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from paths import ROOT, SCALE, inside, new_output, partition_files
N=int(os.environ.get("E12_N", "3"))
assert N in (3,5,10)
sys.path.insert(0,str(BASE/'code'))
import numpy as np
import torch
from pipeline import CheckedEnv,env_config,CommunityTrainingConfig,_make_lr_schedule,PPO,project_allocation
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from stable_baselines3.common.logger import configure

def write(path,value):
 path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_data():
 for name in ('TRAIN','TUNING'):
  m=json.loads((BASE/f'configs/data/{name}.json').read_text())
  for f in m['files']:assert digest(inside(f['path']))==f['sha256']

def config(tuning=False):
 c=env_config(N)
 c.profile_split='TUNING' if tuning else 'TRAIN'
 if tuning:c.price_daily_variation=0.
 return c

class TelemetryEnv(CheckedEnv):
 def __init__(self,config,allocation_decoder='corrected',economic_clip_limit=5.):
  self.economic_clip_limit=float(economic_clip_limit)
  super().__init__(config,allocation_decoder=allocation_decoder)
 def _compute_reward(self,betas,total_cost,baseline_cost,pv_wasted,pv_total):
  reward,info=super()._compute_reward(betas,total_cost,baseline_cost,pv_wasted,pv_total)
  limit=self.economic_clip_limit
  economic=self.config.advantage_weight*float(np.clip(info['advantage_eur'],-limit,limit))
  reward=reward+(economic-info['reward_advantage'])
  info['reward_advantage']=economic;info['reward_total']=float(reward)
  return float(reward),info
 def reset(self,**kwargs):
  obs,info=super().reset(**kwargs);self.high_threshold=float(np.quantile(self._price_expanded,.75));return obs,info
 def step(self,action):
  prop=self._pv_ratios+.6*np.clip(np.asarray(action[:self.config.n_sites],np.float32),-1,1)
  obs,r,t,u,info=super().step(action)
  beta=np.array(info['betas']);soc=np.array([s.battery.SOC_history[-4:] for s in self._sites]);power=np.array([s.battery.Pbat_history[-4:] for s in self._sites])
  info['audit']=dict(cost_eur=info['total_cost']*4,raw_advantage=info['advantage_eur'],advantage_clipped=float(abs(info['advantage_eur'])>self.economic_clip_limit),
   advantage_clipped_at_old5=float(abs(info['advantage_eur'])>5),
   advantage_positive_clip=float(info['advantage_eur']>self.economic_clip_limit),advantage_negative_clip=float(info['advantage_eur']<-self.economic_clip_limit),
   advantage_before_squared=float(info['advantage_eur']**2),advantage_after=float(np.clip(info['advantage_eur'],-self.economic_clip_limit,self.economic_clip_limit)),
   advantage_after_squared=float(np.clip(info['advantage_eur'],-self.economic_clip_limit,self.economic_clip_limit)**2),
   economic_clip_limit=self.economic_clip_limit,
   beta_floor=float(np.mean(beta<=.010001)),projection_l1=float(abs(prop-beta).sum()),projection_centered_l2=float(np.linalg.norm(prop-(prop.sum()-1)/self.config.n_sites-beta)),
   dr=float(np.mean(info['discharge_rates'])),sr=float(np.mean(info['soc_reserves'])),soc=float(soc.mean()),
   discharge_kwh=float(np.maximum(-power,0).sum()*.25/1000),throughput_kwh=float(abs(power).sum()*.25/1000),
   high_price=float(info['price']>=self.high_threshold),constraints=int(np.sum(beta<.01-1e-6)+np.sum(soc<10-1e-6)+np.sum(soc>90+1e-6)+int(abs(beta.sum()-1)>1e-5)))
  return obs,r,t,u,info

class Records:
 def __init__(self,path):self.path=Path(path);self.stream=None;self.writer=None
 def add(self,row):
  if self.stream is None:
   self.stream=self.path.open('x',newline='');self.writer=csv.DictWriter(self.stream,fieldnames=list(row));self.writer.writeheader()
  self.writer.writerow(row);self.stream.flush()
 def close(self):
  if self.stream:self.stream.close()

class Episode:
 def __init__(self):self.count=0;self.sums={};self.beta=np.zeros(N);self.high_count=0;self.high=np.zeros(3)
 def add(self,info,raw,std):
  d=dict(info['audit']);high=d.pop('high_price');self.count+=1;self.beta+=np.array(info['betas'])
  for key in ('reward_advantage','reward_waste','reward_smooth','reward_fairness','reward_total'):d[key]=info[key]
  for i,head in enumerate(('beta','dr','sr')):
   d[f'policy_std_{head}']=float(np.mean(std[i*N:(i+1)*N]));d[f'raw_outside_{head}']=float(np.mean(abs(raw[i*N:(i+1)*N])>1))
  d['raw_outside']=float(np.mean(abs(raw)>1))
  for k,v in d.items():self.sums[k]=self.sums.get(k,0.)+v
  if high:self.high_count+=1;self.high+=np.array([d['sr'],d['soc'],d['discharge_kwh']])
 def finish(self):
  totals={'cost_eur','discharge_kwh','throughput_kwh','constraints','reward_advantage','reward_waste','reward_smooth','reward_fairness','reward_total'}
  row={k:v if k in totals else v/self.count for k,v in self.sums.items()};b=self.beta/self.count
  row.update(decisions=self.count,jain=float(b.sum()**2/(N*np.sum(b*b))),equivalent_full_cycles=row['throughput_kwh']/(2*sum(config().battery_capacities)*config().battery_voltage/1000),high_price_decisions=self.high_count,
   high_price_sr=float(self.high[0]/max(1,self.high_count)),high_price_soc=float(self.high[1]/max(1,self.high_count)),high_price_discharge_kwh=float(self.high[2]))
  return row

class Telemetry(BaseCallback):
 def __init__(self,out):super().__init__();self.out=out;self.episodes=[Episode() for _ in range(8)];self.records=Records(out/'training/episodes.csv');self.finished=0
 def _on_rollout_start(self):
  self.std=self.model.policy.log_std.detach().exp().cpu().numpy()
  write(self.out/'heartbeat.json',dict(timesteps=self.num_timesteps,pid=os.getpid(),phase='training',timestamp=time.time()))
 def _on_step(self):
  for i,(info,raw,done) in enumerate(zip(self.locals['infos'],self.locals['actions'],self.locals['dones'])):
   self.episodes[i].add(info,raw,self.std)
   if done:
    self.finished+=1;self.records.add(dict(timesteps=self.num_timesteps,episode=self.finished,worker=i,**self.episodes[i].finish()));self.episodes[i]=Episode()
  if self.n_calls==1:write(self.out/'startup.json',dict(ready=True,pid=os.getpid(),first_transition=self.num_timesteps))
  return True
 def _on_training_end(self):self.records.close()

def evaluate(model,norm,out,economic_clip_limit):
 norm.training=False;norm.norm_reward=False
 before=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
 seeds=json.loads((BASE/'configs/data/tuning_scenarios.json').read_text())['seeds'];rows=[];detail=Records(out/'evaluation/decisions.csv')
 std=model.policy.log_std.detach().exp().cpu().numpy()
 for seed in seeds:
  env=TelemetryEnv(config(tuning=True),economic_clip_limit=economic_clip_limit);obs,_=env.reset(seed=seed);ep=Episode()
  selected=[p['csv_name'] for p in env._site_profiles]
  expected=json.loads((BASE/f'configs/data/expected_tuning_inputs_N{N}.json').read_text())[str(seed)]
  actual_digest=hashlib.sha256(b''.join(np.asarray(p['load']).tobytes() for p in env._site_profiles)+env._pv_total_profile.tobytes()+env._price_expanded.tobytes()).hexdigest()
  assert actual_digest==expected['input_digest'] and selected==expected['csv_names'],'TUNING input drift'
  for hour in range(168):
   inference_start=time.perf_counter()
   z=norm.normalize_obs(obs)
   with torch.no_grad():raw=model.policy.get_distribution(torch.as_tensor(z[None],dtype=torch.float32)).distribution.mean.cpu().numpy()[0]
   a=np.clip(raw,-1,1).astype(np.float32);inference_seconds=time.perf_counter()-inference_start
   obs,reward,done,trunc,info=env.step(a);info["audit"]["inference_seconds"]=inference_seconds;ep.add(info,raw,std)
   detail.add(dict(scenario_seed=seed,hour=hour,**info['audit'],reward_total=reward,reward_advantage=info['reward_advantage'],reward_waste=info['reward_waste'],reward_smooth=info['reward_smooth'],reward_fairness=info['reward_fairness'],raw_outside=float(np.mean(abs(raw)>1)),policy_std_beta=float(std[:N].mean()),policy_std_dr=float(std[N:2*N].mean()),policy_std_sr=float(std[2*N:].mean())))
   assert done==(hour==167) and not trunc
  rows.append(dict(scenario_seed=seed,**ep.finish()))
  write(out/f'evaluation/scenario_{seed}.json',dict(csv_names=selected,metrics=rows[-1],input_digest=hashlib.sha256(b''.join(np.asarray(p['load']).tobytes() for p in env._site_profiles)+env._pv_total_profile.tobytes()+env._price_expanded.tobytes()).hexdigest()))
  env.close()
 detail.close();rec=Records(out/'evaluation/tuning.csv')
 for row in rows:rec.add(row)
 rec.close()
 np.testing.assert_array_equal(before[0],norm.obs_rms.mean);np.testing.assert_array_equal(before[1],norm.obs_rms.var);assert before[2]==norm.obs_rms.count
 write(out/'evaluation/summary.json',dict(primary_metric='mean_tuning_cost_eur',mean_tuning_cost_eur=float(np.mean([r['cost_eur'] for r in rows])),std_scenario_cost_eur=float(np.std([r['cost_eur'] for r in rows],ddof=1)),normalizer_frozen=True,final_test_used=False,scenarios=seeds))
