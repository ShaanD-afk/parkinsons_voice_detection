from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import stft
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.italian_multitask.audio import read_audio,mono_for_analysis
from src.italian_multitask.segmentation import rms_vad

DATA=Path('/Users/Shaan/Desktop/datasets/italian')
OUT=ROOT/'results/italian_research_phase2'

def make_d12():
    d=pd.read_csv(OUT/'d1d2/recording_characterization.csv')
    folder=OUT/'d1d2/diagnostic_plots';folder.mkdir(parents=True,exist_ok=True)
    for r in d.itertuples(index=False):
        x,sr,_=read_audio(DATA/r.relative_path);a,_=mono_for_analysis(x);dur=len(a)/sr;t=np.arange(len(a))/sr
        fig,axs=plt.subplots(3,1,figsize=(12,7),constrained_layout=True)
        stride=max(1,len(a)//50000);axs[0].plot(t[::stride],a[::stride],lw=.4);axs[0].set_ylabel('Amplitude')
        f,tt,z=stft(a,sr,nperseg=int(.025*sr),noverlap=int(.015*sr),boundary=None)
        db=20*np.log10(np.maximum(abs(z),1e-7));axs[1].pcolormesh(tt,f,db,shading='auto',cmap='magma',vmin=-90,vmax=-15);axs[1].set_ylim(0,min(6000,sr/2));axs[1].set_ylabel('Hz')
        win=int(.025*sr);hop=int(.01*sr);levels=[]
        for i in range(0,max(1,len(a)-win),hop):levels.append(np.sqrt(np.mean(a[i:i+win]**2)))
        env=np.asarray(levels);et=np.arange(len(env))*.01;th=float(r.event_threshold_rms)
        axs[2].plot(et,env,lw=.8);axs[2].axhline(th,color='red',ls='--',label='exploratory adaptive event threshold')
        if r.candidate_event_times_s:
            for v in str(r.candidate_event_times_s).split(';'):
                if v: axs[2].axvline(float(v),color='green',alpha=.5,lw=.6)
        axs[2].set_ylabel('RMS');axs[2].set_xlabel('Time (s)');axs[2].legend(fontsize=7)
        for ax in axs:ax.set_xlim(0,dur);ax.grid(alpha=.15)
        fig.suptitle(f'{r.task_code} | {r.group} | {r.participant_folder} | {sr} Hz | candidate peaks={r.candidate_energy_peaks_n} (not syllable labels)')
        fig.savefig(folder/(Path(r.relative_path).stem+'.png'),dpi=110);plt.close(fig)
    for code in ['D1','D2']:
        files=sorted(folder.glob('*.png'))
        # Generate code-specific contact sheets from the task rows, matching original file stems.
        stems=set(Path(p).stem for p in d[d.task_code.eq(code)].relative_path)
        selected=[p for p in files if p.stem in stems]
        ncol=3;nrow=int(np.ceil(len(selected)/ncol))
        fig,axs=plt.subplots(nrow,ncol,figsize=(18,3.2*nrow),squeeze=False)
        for ax in axs.flat:ax.axis('off')
        for ax,p in zip(axs.flat,selected):ax.imshow(mpimg.imread(p));ax.set_title(p.stem,fontsize=7);ax.axis('off')
        fig.suptitle(f'{code} all-recording waveform/spectrogram/envelope diagnostics')
        fig.tight_layout();fig.savefig(OUT/f'd1d2/{code.lower()}_all_recordings_contact_sheet.png',dpi=70);plt.close(fig)

def make_reading_vad():
    q=pd.read_csv(OUT/'review/human_review_queue.csv')
    q=q[q.source.astype(str).str.contains('connected_speech_stratified_threshold_review')]
    # Include all selected strata (one/two recordings per group × rate × task).
    folder=OUT/'connected_speech/stratified_vad_review_plots';folder.mkdir(parents=True,exist_ok=True)
    for r in q.itertuples(index=False):
        a0,sr,_=read_audio(DATA/r.relative_path);a,_=mono_for_analysis(a0);dur=len(a)/sr;t=np.arange(len(a))/sr
        fig,axs=plt.subplots(3,1,figsize=(12,7),constrained_layout=True)
        stride=max(1,len(a)//50000);axs[0].plot(t[::stride],a[::stride],lw=.4);axs[0].set_ylabel('Amplitude')
        f,tt,z=stft(a,sr,nperseg=int(.025*sr),noverlap=int(.015*sr),boundary=None)
        db=20*np.log10(np.maximum(abs(z),1e-7));axs[1].pcolormesh(tt,f,db,shading='auto',cmap='magma',vmin=-90,vmax=-15);axs[1].set_ylim(0,min(6000,sr/2));axs[1].set_ylabel('Hz')
        lev,_,_=rms_vad(a,sr);lt=np.arange(len(lev))*.01
        for dbbelow,color in [(-30,'green'),(-35,'orange'),(-40,'red')]:
            p95=np.percentile(lev,95);thr=max(1e-7,p95*10**(dbbelow/20));axs[2].plot(lt,lev,lw=.5,color='gray')
            axs[2].axhline(thr,color=color,ls='--',label=f'{dbbelow} dB: {thr:.4g}')
        axs[2].set_ylabel('RMS');axs[2].set_xlabel('Time (s)');axs[2].legend(fontsize=7,ncol=3)
        for ax in axs:ax.set_xlim(0,dur);ax.grid(alpha=.15)
        fig.suptitle(f'{r.task_code} | {r.group} | {r.participant_folder} | VAD threshold sensitivity')
        fig.savefig(folder/(r.review_id+'.png'),dpi=110);plt.close(fig)

if __name__=='__main__':
    make_d12();make_reading_vad()
