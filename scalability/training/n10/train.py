def vec_for(spec):
 return S.make_vec_env(S.TelemetryEnv,n_envs=8,seed=spec['seed'],vec_env_cls=S.SubprocVecEnv,env_kwargs={'config':S.config(),'economic_clip_limit':spec['economic_clip_limit']},vec_env_kwargs={'start_method':'fork'})

"""Fresh training adapter for the published configuration only."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--size",type=int,choices=[10],required=True)
 parser.add_argument("--seed",type=int,choices=[42,142,242],required=True)
 parser.add_argument("--output",required=True)
 args=parser.parse_args()
 base=Path(__file__).resolve().parents[2]
 sys.path.insert(0,str(base))
 from paths import new_output
 spec=json.loads((base/f"configs/training/N{args.size}_seed{args.seed}.json").read_text())
 if spec["N"]!=args.size or spec["seed"]!=args.seed or not spec["fresh"] or spec["initial_steps"]!=0:
  raise ValueError("Only the selected fresh configuration is supported")
 os.environ["E12_N"]=str(args.size)
 global S
 import support as S
 out=new_output(args.output)
 for folder in ("training","models","evaluation"): (out/folder).mkdir()
 job=f"N{args.size}_seed{args.seed}"
 S.verify_data();S.torch.set_num_threads(1);cfg=spec["ppo"];norm=None
 try:
  vec=vec_for(spec);norm=S.VecNormalize(vec,norm_obs=True,norm_reward=True,clip_obs=10.,clip_reward=10.)
  model=S.PPO(cfg['policy'],norm,learning_rate=S._make_lr_schedule(cfg['learning_rate'],cfg['lr_schedule']),n_steps=cfg['n_steps'],batch_size=cfg['batch_size'],n_epochs=cfg['n_epochs'],gamma=cfg['gamma'],gae_lambda=cfg['gae_lambda'],clip_range=cfg['clip_range'],ent_coef=cfg['ent_coef'],vf_coef=cfg['vf_coef'],max_grad_norm=cfg['max_grad_norm'],policy_kwargs=cfg['policy_kwargs'],target_kl=cfg['target_kl'],seed=spec['seed'],device='cpu')
  S.write(out/'initialization.json',dict(from_scratch=True,initial_steps=model.num_timesteps))
  start_steps=model.num_timesteps;target=spec['target_total_steps'];S.write(out/'training/config.json',cfg);S.write(out/'training/environment.json',dict(**S.asdict(S.config()),economic_clip_limit=spec['economic_clip_limit']))
  model.set_logger(S.configure(str(out/'training'),['stdout','csv']))
  callback=S.CallbackList([S.Telemetry(out),S.CheckpointCallback(save_freq=6250,save_path=str(out/'models/checkpoints'),name_prefix='ppo_community',save_vecnormalize=True)])
  start=time.perf_counter()
  if model.num_timesteps<target:model.learn(total_timesteps=target-model.num_timesteps,reset_num_timesteps=False,callback=callback)
  model.save(out/'models/ppo_community');norm.save(str(out/'models/vecnorm_community_stats.pkl'))
  S.write(out/'training/completion.json',dict(initial_steps=start_steps,target_steps=target,actual_steps=model.num_timesteps,training_seconds=time.perf_counter()-start))
  S.write(out/'heartbeat.json',dict(timesteps=model.num_timesteps,pid=S.os.getpid(),phase='tuning',timestamp=time.time()))
  S.evaluate(model,norm,out,spec['economic_clip_limit'])
  result=dict(job=job,attempt='001',actual_steps=model.num_timesteps,model=str((out/'models/ppo_community.zip').relative_to(S.ROOT)),vecnormalize=str((out/'models/vecnorm_community_stats.pkl').relative_to(S.ROOT)),evaluation=str((out/'evaluation/tuning.csv').relative_to(S.ROOT)),final_test_used=False)

  S.write(out/'result.json',result)
 finally:
  if norm is not None:norm.close()

if __name__=="__main__":main()
