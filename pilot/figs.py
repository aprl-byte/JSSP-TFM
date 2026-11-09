import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, pandas as pd
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
cols=['#20808D','#A84B2F','#1B474D','#FFC553','#944454','#848456']
S=json.load(open('ft06_sched.json'))
fig,axs=plt.subplots(2,1,figsize=(8,5.2),sharex=True)
for ax,(k,title) in zip(axs,[('MWKR','Regla MWKR (Cmax = 67)'),('CPSAT','CP-SAT, solución óptima (Cmax = 55)')]):
    for i,kk,m,s,p in S[k]:
        ax.barh(m,p,left=s,color=cols[i],edgecolor='white',height=0.8); ax.text(s+p/2,m,f'J{i+1}',ha='center',va='center',color='white' if i!=3 else '#28251D',fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels([f'M{j+1}' for j in range(6)]); ax.invert_yaxis(); ax.set_title(title,loc='left',fontsize=10)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
axs[1].set_xlabel('Tiempo'); plt.tight_layout(); plt.savefig('fig_gantt.png',dpi=200)
d=json.load(open('exp.json'))['rand10x10']; df=pd.DataFrame(d); rules=['FIFO','SPT','LPT','MWKR','LWKR','MOPNR']
g=[100*(df[r]-df.ref)/df.ref for r in rules]
fig,ax=plt.subplots(figsize=(7,3.6)); b=ax.boxplot(g,tick_labels=rules,patch_artist=True,widths=0.55)
for p in b['boxes']: p.set_facecolor('#BCE2E7'); p.set_edgecolor('#1B474D')
for p in b['medians']: p.set_color('#A84B2F')
ax.set_ylabel('Gap respecto al óptimo (%)'); [ax.spines[s].set_visible(False) for s in ['top','right']]; ax.grid(axis='y',alpha=.3)
plt.tight_layout(); plt.savefig('fig_gap.png',dpi=200)
steps=['Instancias\nsintéticas','Soluciones de\nreferencia\n(CP-SAT)','Etiquetado\nde decisiones','Entrenamiento\ndel clasificador','Integración\ncomo regla\nde prioridad','Evaluación\nfrente a PDR\n(Taillard, DMU)']
fig,ax=plt.subplots(figsize=(9,1.9)); ax.axis('off'); ax.set_xlim(0,6); ax.set_ylim(0,1)
for k,t in enumerate(steps):
    ax.add_patch(FancyBboxPatch((k+0.06,0.12),0.82,0.76,boxstyle='round,pad=0.02',fc='#F9F8F5',ec='#01696F',lw=1.2)); ax.text(k+0.47,0.5,t,ha='center',va='center',fontsize=7.5,color='#28251D')
    if k<5: ax.annotate('',xy=(k+1.05,0.5),xytext=(k+0.9,0.5),arrowprops=dict(arrowstyle='->',color='#01696F'))
plt.savefig('fig_flujo.png',dpi=200,bbox_inches='tight')
