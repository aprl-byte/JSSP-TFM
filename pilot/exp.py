import numpy as np, json, time
from ortools.sat.python import cp_model
ft06=[[(2,1),(0,3),(1,6),(3,7),(5,3),(4,6)],[(1,8),(2,5),(4,10),(5,10),(0,10),(3,4)],[(2,5),(3,4),(5,8),(0,9),(1,1),(4,7)],[(1,5),(0,5),(2,5),(3,3),(4,8),(5,9)],[(2,9),(1,3),(4,5),(5,4),(0,3),(3,1)],[(1,3),(3,3),(5,9),(0,10),(4,4),(2,1)]]
def dispatch(J,rule,rng=None):
    # Giffler-Thompson: planes activos; la regla elige dentro del conjunto de conflicto
    n=len(J); m=1+max(mm for j in J for mm,_ in j)
    nxt=[0]*n; jr=[0]*n; mr=[0]*m; sched=[]
    rem=[sum(p for _,p in j) for j in J]
    while any(nxt[i]<len(J[i]) for i in range(n)):
        c=[i for i in range(n) if nxt[i]<len(J[i])]
        est={i:max(jr[i],mr[J[i][nxt[i]][0]]) for i in c}
        ect={i:est[i]+J[i][nxt[i]][1] for i in c}
        istar=min(c,key=lambda i:ect[i]); ms=J[istar][nxt[istar]][0]
        C=[i for i in c if J[i][nxt[i]][0]==ms and est[i]<ect[istar]]
        def key(i):
            mm,p=J[i][nxt[i]]
            return ({'SPT':p,'LPT':-p,'MWKR':-rem[i],'LWKR':rem[i],'MOPNR':-(len(J[i])-nxt[i]),'FIFO':est[i],'RAND':rng.random() if rng else 0}[rule],i)
        i=min(C,key=key); mm,p=J[i][nxt[i]]; s=est[i]
        sched.append((i,nxt[i],mm,s,p)); jr[i]=mr[mm]=s+p; rem[i]-=p; nxt[i]+=1
    return max(jr),sched
def cpsat(J,tl=10):
    md=cp_model.CpModel(); H=sum(p for j in J for _,p in j); iv={}; ends=[]; mach={}
    for i,j in enumerate(J):
        prev=None
        for k,(mm,p) in enumerate(j):
            s=md.NewIntVar(0,H,''); e=md.NewIntVar(0,H,''); x=md.NewIntervalVar(s,p,e,''); mach.setdefault(mm,[]).append(x); iv[(i,k)]=(s,e,mm,p)
            if prev is not None: md.Add(s>=prev)
            prev=e
        ends.append(prev)
    for xs in mach.values(): md.AddNoOverlap(xs)
    mk=md.NewIntVar(0,H,''); md.AddMaxEquality(mk,ends); md.Minimize(mk)
    so=cp_model.CpSolver(); so.parameters.max_time_in_seconds=tl; so.parameters.num_workers=8; t=time.time(); st=so.Solve(md)
    sched=[(i,k,v[2],so.Value(v[0]),v[3]) for (i,k),v in iv.items()]
    return so.Value(mk), st==cp_model.OPTIMAL, time.time()-t, sched
rules=['FIFO','SPT','LPT','MWKR','LWKR','MOPNR']
out={'ft06':{}}
for r in rules: out['ft06'][r]=dispatch(ft06,r)[0]
opt=cpsat(ft06); out['ft06']['CPSAT']=opt[0]; out['ft06']['opt']=opt[1]
json.dump({'SPT':dispatch(ft06,'SPT')[1],'MWKR':dispatch(ft06,'MWKR')[1],'CPSAT':opt[3]},open('ft06_sched.json','w'))
rng=np.random.default_rng(42); res=[]
for t in range(30):
    J=[[(int(mm),int(rng.integers(1,100))) for mm in rng.permutation(10)] for _ in range(10)]
    ref=cpsat(J,10); row={'inst':t,'ref':ref[0],'opt':ref[1],'t_ref':ref[2]}
    for r in rules:
        t0=time.perf_counter(); row[r]=dispatch(J,r)[0]; row['t_'+r]=time.perf_counter()-t0
    res.append(row)
out['rand10x10']=res; json.dump(out,open('exp.json','w'),indent=1)
print(out['ft06'])
import pandas as pd; df=pd.DataFrame(res)
for r in rules: df['gap_'+r]=100*(df[r]-df.ref)/df.ref
print(df[['gap_'+r for r in rules]].describe().round(1)); print('opt all',df.opt.all(), 'tref mean',df.t_ref.mean())
print('best rule counts', df[['gap_'+r for r in rules]].idxmin(axis=1).value_counts())
print('mean t per rule ms', {r: round(1000*df['t_'+r].mean(),3) for r in rules})
