"""Evaluate only the five published scalability methods."""
import argparse,json,os,sys
from pathlib import Path

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--sizes",nargs="+",type=int,choices=[3,5,10],default=[3,5,10])
 parser.add_argument("--methods",nargs="+",choices=["uniform","proportional","rule_based","ppo","mpc"],default=["uniform","proportional","rule_based","ppo","mpc"])
 parser.add_argument("--seeds",nargs="+",type=int,choices=[42,142,242],default=[42,142,242])
 parser.add_argument("--scenarios",nargs="+",type=int,choices=range(1,11),default=list(range(1,11)))
 parser.add_argument("--max-decisions",type=int,default=168)
 parser.add_argument("--output",required=True)
 args=parser.parse_args()
 for values in (args.sizes,args.methods,args.seeds,args.scenarios):
  if len(values)!=len(set(values)): parser.error("Duplicate selections")
 if not 1<=args.max_decisions<=168: parser.error("--max-decisions must be 1..168")
 scale=Path(__file__).resolve().parents[1];sys.path.insert(0,str(scale))
 from paths import new_output
 # Set thread limits before scientific imports.
 for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
  os.environ[name]="1"
 import common as C
 C.torch.set_num_threads(1)
 out=new_output(args.output)
 selection=dict(vars(args),output=str(out.relative_to(scale.parent)))
 (out/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
 methods=[m for m in args.methods if m!="ppo"]+([f"ppo_{s}" for s in args.seeds] if "ppo" in args.methods else [])
 for n in args.sizes:
  for method in methods:
   policy=C.FrozenPolicy(n,int(method.split("_")[1])) if method.startswith("ppo_") else None
   for week in args.scenarios:C.run_week(out,n,week,method,policy,max_decisions=args.max_decisions)
   if policy:policy.check_frozen()

if __name__=="__main__":main()
