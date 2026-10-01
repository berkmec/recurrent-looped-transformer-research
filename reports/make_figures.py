"""Standalone research figures from retained per-example evidence."""
from pathlib import Path
import os,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'audit/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelsize':11,'axes.titlesize':13,
                     'figure.dpi':130,'savefig.dpi':180})
FIG=ROOT/'reports/figures';FIG.mkdir(parents=True,exist_ok=True)
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']


def save(fig,name):
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def learning():
    arms=['forward','bidir','method','combined']
    labels=['İleri yön','Çift yön','Çift yön +\nara gözetim','Ara gözetim +\ngeri besleme']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharey=True)
    for ax,split,title in zip(axes,['iid','depth8'],['Eğitim benzeri yeni problemler','Görülmemiş bağımlılık derinliği: 8']):
        for i,arm in enumerate(arms):
            values=[]
            for seed in [17,43,91]:
                file=ROOT/'learning/results'/f'pred-{arm}-{seed}-600-{split}-retain.npz'
                with np.load(file) as data:values.append(100*np.mean(np.abs(data['predictions'][:,-1]-data['targets'][:,-1])<=.1))
            ax.bar(i,np.mean(values),width=.62,color=COLORS[i],alpha=.72,zorder=2)
            ax.scatter(i+np.array([-.09,0,.09]),values,c='black',s=22,zorder=3)
            ax.text(i,max(values)+3,f'{np.mean(values):.1f}%',ha='center',fontsize=11)
        ax.set_xticks(range(4),labels);ax.set_ylim(0,109);ax.set_title(title,pad=13)
        ax.grid(axis='y',alpha=.18,zorder=0);ax.set_xlabel('Aynı sayıda eğitim güncellemesi')
    axes[0].set_ylabel('Son cevap doğruluğu (%)  |  hata ≤ 0,1')
    fig.suptitle('Başlık ağı ve ara gözetim: fayda problem derinliğine bağlı',y=1.06,fontsize=15)
    fig.text(.5,-.05,'Her sütun üç eğitimin ortalaması; siyah noktalar tekil eğitimler. Her bölümde 512 ortak test problemi.',ha='center',fontsize=10)
    fig.tight_layout();save(fig,'learning-generalization')


def connected():
    summary=json.loads((ROOT/'integration/connected/summary.json').read_text())
    names=['adaptive-replay-seen','fixed-replay-seen','adaptive-compact-seen','adaptive-replay-heldout']
    labels=['Tam geçmiş\nuyarlanır durma','Tam geçmiş\nsabit 8 tur','Güncel özet\nuyarlanır durma','Yeni cümle düzeni\ntam geçmiş']
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),sharey=True)
    for ax,split,title in zip(axes,['iid','depth8'],['Eğitim benzeri yeni problemler','Görülmemiş bağımlılık derinliği: 8']):
        candidate=[];yield_=[]
        for name in names:
            rows=[r for r in summary['rows'] if r['split']==split and r['condition']==name]
            candidate.append(100*np.mean([r['diagnostic_candidate_accuracy'] for r in rows]))
            yield_.append(100*np.mean([r['certified_source_correct_yield'] for r in rows]))
        x=np.arange(4)
        ax.bar(x-.17,candidate,.32,color=COLORS[0],alpha=.76,label='Doğru sayısal aday',zorder=2)
        ax.bar(x+.17,yield_,.32,color=COLORS[2],alpha=.86,label='Doğru ve doğrulamadan geçen cevap',zorder=2)
        for i,(a,b) in enumerate(zip(candidate,yield_)):
            ax.text(i-.17,a+2,f'{a:.1f}',ha='center',fontsize=9)
            ax.text(i+.17,b+2,f'{b:.1f}',ha='center',fontsize=9)
        ax.set_xticks(x,labels);ax.set_ylim(0,108);ax.set_title(title,pad=13);ax.grid(axis='y',alpha=.18,zorder=0)
    axes[0].set_ylabel('Amaçlanan son soruların yüzdesi (%)')
    handles,labels_=axes[0].get_legend_handles_labels();fig.legend(handles,labels_,loc='lower center',bbox_to_anchor=(.5,-.055),ncol=2,frameon=False)
    fig.suptitle('Kaynağa göre doğruluk ve doğrulamadan geçen cevap kapsamı',y=1.04,fontsize=15)
    fig.tight_layout();save(fig,'connected-quality-coverage')


def core():
    m=json.loads((ROOT/'core/results/metrics.json').read_text())
    assert m['status']=='COMPLETE' and len(m['evaluation'])==21
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharey=True)
    lengths=np.array([32,64,128,256,512,1024])
    for ax,task,title in zip(axes,['parity','five'],['Parite durumu','Beş durumlu sıralı işlem']):
        for index,mode in enumerate(['rlt','nofeedback','tbptt'] if task=='parity' else ['rlt','nofeedback']):
            rows=[r for r in m['evaluation'] if r['condition']['task']==task and r['condition']['width']==32 and r['condition']['mode']==mode]
            values=np.array([[v['final_accuracy'] for v in r['long_horizon']] for r in sorted(rows,key=lambda r:r['condition']['seed'])])*100
            label={'rlt':'RLT','nofeedback':'Gizli durum geri beslemesi kapalı','tbptt':'RLT · kesilmiş gradyan'}[mode]
            ax.plot(lengths,values.mean(0),marker='o',color=COLORS[index],label=label,lw=2)
            ax.fill_between(lengths,values.min(0),values.max(0),color=COLORS[index],alpha=.12)
        ax.axhline(50 if task=='parity' else 20,color='gray',ls=':',lw=1.1,label='Şans düzeyi')
        ax.axvline(32,color='gray',lw=.7,alpha=.5);ax.set_xscale('log',base=2);ax.set_xticks(lengths,[str(n) for n in lengths])
        ax.set_ylim(0,104);ax.set_title(title);ax.set_xlabel('Tüketilen işlem/token sayısı');ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Son durum doğruluğu (%)')
    handles,labels_=axes[0].get_legend_handles_labels();fig.legend(handles,labels_,loc='lower center',bbox_to_anchor=(.5,-.09),ncol=2,frameon=False)
    fig.suptitle('32 adımda eğitimden sonra ölçülen uzun ufuk',y=1.04,fontsize=15)
    fig.text(.5,-.16,'Genişlik 32; her adımda hedef verilen eğitim. Paritede mavi ve yeşil çizgiler üst üste.\nÇizgiler üç eğitimin ortalaması; gölge en düşük–en yüksek sonuç, güven aralığı değil.',ha='center',fontsize=10)
    fig.tight_layout();save(fig,'core-long-horizon')


def scale():
    m=json.loads((ROOT/'integration/scale/metrics.json').read_text())
    assert m['status']=='COMPLETE'
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),sharey=True)
    for index,seed in enumerate([17,43,91]):
        e=next(r for r in m['evaluation'] if r['seed']==seed and r['mode']=='rlt')
        for ax,family,key in [(axes[0],'long_horizon','length'),(axes[1],'neutral_slots','slots')]:
            rows=e[family]
            ax.plot([r[key] for r in rows],[100*r['final_accuracy'] for r in rows],
                    marker='o',color=COLORS[index],label=f'Eğitim tohumu {seed}',lw=2)
    axes[0].set_xscale('log',base=2)
    axes[0].set_xticks([32,64,128,256,512,1024],['32','64','128','256','512','1024'])
    axes[0].set_title('Yeni rastgele işlem dizileri')
    axes[0].set_xlabel('Dizi uzunluğu · eğitimde 32')
    axes[1].set_xticks([0,8,32,128]);axes[1].set_title('Aynı 32 işleme etkisiz son ek')
    axes[1].set_xlabel('Eklenen kimlik işlemi sayısı · k = 0')
    for ax in axes:
        ax.axhline(50,color='gray',ls=':',lw=1.1);ax.set_ylim(0,106);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Son durum doğruluğu (%)')
    handles,labels_=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels_,loc='lower center',bbox_to_anchor=(.5,-.025),ncol=3,frameon=False)
    fig.suptitle('1,51 milyon parametreli RLT: eğitim tekrarı ve işlem düzeni belirleyici',y=1.04,fontsize=14)
    fig.text(.5,-.09,'Her noktada 512 örnek. Sağ panel aynı başlangıç dizilerini kullanır; uzunluk testiyle aynı örnekler değildir.',ha='center',fontsize=10)
    fig.tight_layout();save(fig,'scale-seeds-and-neutral')


if __name__=='__main__':
    learning()
    if '--scale' in sys.argv or '--all' in sys.argv:scale()
    if '--all' in sys.argv:connected();core()
