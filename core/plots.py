"""Figures from frozen evaluation results; seed ranges, not pooled-prefix CIs."""
import json,os
from pathlib import Path
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 root=Path(__file__).parent;(root/'results'/'plot-environment.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,numpy=np.__version__),indent=2));d=json.loads((root/'results'/'metrics.json').read_text());plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained');colors=dict(rlt='#176c47',nofeedback='#ba4424',tbptt='#5351a4')
 for ax,(task,width) in zip(axs,[('parity',32),('five',32),('parity',48)]):
  for mode,color in colors.items():
   rows=[e for e in d['evaluation'] if e['condition']['task']==task and e['condition']['width']==width and e['condition']['mode']==mode]
   if not rows:continue
   xx=[r['length'] for r in rows[0]['long_horizon']];yy=np.array([[r['final_accuracy'] for r in e['long_horizon']] for e in rows])*100;ax.plot(xx,yy.mean(0),'-o',label=mode,color=color);ax.fill_between(xx,yy.min(0),yy.max(0),alpha=.12,color=color)
  ax.axhline(50 if task=='parity' else 20,color='gray',ls=':',label='uniform guess');ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256,512,1024],labels=['32','64','128','256','512','1024']);ax.set_ylim(-2,102);ax.set_title(f'{task}, width {width}');ax.set_xlabel('Consumed tokens');ax.set_ylabel('Final-state accuracy (%)');ax.legend(fontsize=8)
 fig.suptitle('Training length 32; mean and min–max across 3 seeds,512 test examples per seed');fig.savefig(root/'results'/'accuracy.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for ax,task in zip(axs,['parity','five']):
  for mode,color in colors.items():
   rows=[e for e in d['evaluation'] if e['condition']['task']==task and e['condition']['width']==32 and e['condition']['mode']==mode]
   if not rows:continue
   xx=np.array([[r['seconds'] for r in e['neutral_slots']] for e in rows]).mean(0);yy=np.array([[r['final_accuracy'] for r in e['neutral_slots']] for e in rows]).mean(0)*100;ax.plot(xx,yy,'-o',color=color,label=mode)
   for x,y,k in zip(xx,yy,[0,8,32,128]):ax.annotate(str(k),(x,y),fontsize=8,xytext=(3,5),textcoords='offset points')
  ax.set_title(task);ax.set_xlabel('Seconds to evaluate 512 examples (CPU shared load)');ax.set_ylabel('Final-state accuracy (%)');ax.set_ylim(-2,105);ax.legend(fontsize=8)
 fig.suptitle('Same task, added neutral tokens; labels show added slots, k=0');fig.savefig(root/'results'/'neutral-quality-time.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()

