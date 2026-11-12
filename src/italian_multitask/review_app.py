from __future__ import annotations

"""Local macOS Tk review tool. Launch with python -m src.italian_multitask.review_app."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import parselmouth
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import stft

from .audio import mono_for_analysis, read_audio
from .review_materials import refresh_review_summary


class ReviewApp:
    def __init__(self, root, dataset_root: Path, experiment_root: Path):
        self.root=root; self.dataset_root=dataset_root; self.experiment_root=experiment_root
        self.queue_path=experiment_root/"review/human_review_queue.csv"
        self.annotation_path=experiment_root/"review/manual_annotations.csv"
        self.queue=pd.read_csv(self.queue_path).fillna("")
        self.annotations=pd.read_csv(self.annotation_path).fillna("")
        refresh_review_summary(experiment_root)
        self.index=0; self.process=None; self.vars={}; self.comment=tk.StringVar(); self.reviewer=tk.StringVar(); self.manual_count=tk.StringVar()
        root.title("ItalianPVS Audio Review")
        root.geometry("1400x950")
        self.title=ttk.Label(root,text="",font=("Helvetica",14,"bold")); self.title.pack(anchor="w",padx=10,pady=4)
        self.meta=ttk.Label(root,text="",wraplength=1350); self.meta.pack(anchor="w",padx=10)
        self.fig,self.axes=plt.subplots(3,1,figsize=(13,6.6),constrained_layout=True)
        self.canvas=FigureCanvasTkAgg(self.fig,master=root); self.canvas.get_tk_widget().pack(fill="both",expand=True)
        ctr=ttk.Frame(root); ctr.pack(fill="x",padx=10,pady=4)
        ttk.Button(ctr,text="◀ Previous",command=lambda:self.move(-1)).pack(side="left")
        ttk.Button(ctr,text="Next ▶",command=lambda:self.move(1)).pack(side="left",padx=4)
        ttk.Button(ctr,text="Play original audio",command=self.play).pack(side="left",padx=8)
        ttk.Button(ctr,text="Stop",command=self.stop).pack(side="left")
        ttk.Button(ctr,text="Save annotation",command=self.save).pack(side="right")
        form=ttk.Frame(root); form.pack(fill="x",padx=10,pady=6)
        choices={"acceptable_recording":["yes","no","uncertain"],
                 "suspected_recording_problem":["none","clipping","noise","truncation","wrong_task","other"],
                 "segmentation_correct":["yes","no","not_applicable","uncertain"],
                 "pitch_tracking_plausible":["yes","no","not_applicable","uncertain"],
                 "formants_plausible":["yes","no","not_applicable","uncertain"],
                 "uncertain":["no","yes"]}
        for i,(field,values) in enumerate(choices.items()):
            ttk.Label(form,text=field.replace("_"," ").title()+":").grid(row=i//3,column=(i%3)*2,sticky="w",padx=3,pady=3)
            v=tk.StringVar(); self.vars[field]=v
            cb=ttk.Combobox(form,textvariable=v,values=values,state="readonly",width=18); cb.grid(row=i//3,column=(i%3)*2+1,sticky="w",padx=3)
        ttk.Label(form,text="Manual event onsets (s, comma-separated; D1/D2):").grid(row=2,column=0,sticky="w",padx=3)
        self.onsets_widget=tk.Text(form,height=2,width=60); self.onsets_widget.grid(row=2,column=1,columnspan=2,sticky="ew")
        ttk.Label(form,text="Manual syllable count:").grid(row=2,column=3,sticky="w",padx=3)
        ttk.Entry(form,textvariable=self.manual_count,width=8).grid(row=2,column=4,sticky="w")
        ttk.Label(form,text="Reviewer:").grid(row=3,column=0,sticky="w",padx=3)
        ttk.Entry(form,textvariable=self.reviewer,width=22).grid(row=3,column=1,sticky="w")
        ttk.Label(form,text="Comments:").grid(row=4,column=0,sticky="nw",padx=3)
        self.comment_widget=tk.Text(form,height=3,width=110); self.comment_widget.grid(row=4,column=1,columnspan=5,sticky="ew")
        self.render()

    def row(self): return self.queue.iloc[self.index]

    def load_annotation(self):
        r=self.row(); hit=self.annotations[(self.annotations.review_id.astype(str)==str(r.review_id)) & (self.annotations.relative_path==r.relative_path)]
        rec=hit.iloc[-1].to_dict() if len(hit) else {}
        for k,v in self.vars.items(): v.set(str(rec.get(k,"")))
        self.reviewer.set(str(rec.get("reviewer_id","")))
        self.manual_count.set(str(rec.get("manual_syllable_count","")))
        self.onsets_widget.delete("1.0","end"); self.onsets_widget.insert("1.0",str(rec.get("manual_event_onsets_s","")))
        self.comment_widget.delete("1.0","end"); self.comment_widget.insert("1.0",str(rec.get("reviewer_comments","")))

    def render(self):
        r=self.row(); path=self.dataset_root/str(r.relative_path)
        self.title.config(text=f"{self.index+1}/{len(self.queue)}  {r.review_id}  [{r.priority}] {r.task_code}")
        self.meta.config(text=f"{r.group} | {r.participant_folder} | {r.sample_rate_hz} Hz\nFlags: {r.reason}\nFile: {r.relative_path}")
        for ax in self.axes: ax.clear()
        try:
            x,sr,_=read_audio(path); audio,_=mono_for_analysis(x); duration=len(audio)/sr
            t=np.arange(len(audio))/sr
            stride=max(1,len(t)//30000); self.axes[0].plot(t[::stride],audio[::stride],lw=.45,color="#315a7d")
            self.axes[0].set_ylabel("Amplitude"); self.axes[0].set_xlim(0,duration); self.axes[0].grid(alpha=.2)
            f,tt,z=stft(audio,sr,nperseg=int(.025*sr),noverlap=int(.015*sr),boundary=None)
            db=20*np.log10(np.maximum(np.abs(z),1e-7))
            self.axes[1].pcolormesh(tt,f,db,shading="auto",cmap="magma",vmin=-90,vmax=-15)
            self.axes[1].set_ylim(0,min(5000,sr/2)); self.axes[1].set_ylabel("Frequency (Hz)")
            sound=parselmouth.Sound(audio,sampling_frequency=sr)
            pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=75,pitch_ceiling=500)
            f0=pitch.selected_array["frequency"]; pt=pitch.xs(); voiced=np.isfinite(f0)&(f0>0)
            self.axes[1].plot(pt[voiced],f0[voiced],"c.",ms=2,label="F0 (75–500 Hz)")
            form=sound.to_formant_burg(time_step=.01,max_number_of_formants=5,maximum_formant=5000,window_length=.025,pre_emphasis_from=50)
            for idx,color in zip([1,2,3],["lime","cyan","white"]):
                times=np.arange(.02,duration,.02); vals=[]
                for ti in times:
                    try: vals.append(form.get_value_at_time(idx,float(ti)))
                    except Exception: vals.append(np.nan)
                vals=np.asarray(vals); good=np.isfinite(vals)&(vals>0)&(vals<sr/2)
                self.axes[1].plot(times[good],vals[good],".",ms=1.5,color=color,alpha=.8)
            self.axes[1].legend(loc="upper right",fontsize=7)
            levels=[]; step=int(.01*sr); win=int(.025*sr)
            for i in range(0,max(1,len(audio)-win),step): levels.append(np.sqrt(np.mean(audio[i:i+win]**2)))
            lt=np.arange(len(levels))*.01; lv=np.asarray(levels)
            p95=np.quantile(lv,.95) if len(lv) else 0; active=lv>=max(1e-7,p95*10**(-35/20))
            self.axes[2].plot(lt,20*np.log10(np.maximum(lv,1e-7)),color="black",lw=.6)
            self.axes[2].axhline(20*np.log10(max(1e-7,p95*10**(-35/20))),color="red",ls="--",lw=.8,label="VAD −35 dB below p95")
            self.axes[2].fill_between(lt,[-100]*len(lt),[-90]*len(lt),where=active,color="green",alpha=.5,step="pre",label="active frames")
            self.axes[2].set_ylabel("RMS (dB)"); self.axes[2].set_xlabel("Time (s)"); self.axes[2].set_xlim(0,duration); self.axes[2].legend(fontsize=7); self.axes[2].grid(alpha=.2)
            self.fig.suptitle("Waveform | spectrogram with Praat F0 and Burg formants | RMS/voicing diagnostic")
        except Exception as e:
            self.axes[0].text(.05,.5,f"Could not render audio: {e}",transform=self.axes[0].transAxes)
        self.canvas.draw_idle(); self.load_annotation()

    def move(self,step):
        self.stop(); self.index=(self.index+step)%len(self.queue); self.render()

    def play(self):
        self.stop(); path=self.dataset_root/str(self.row().relative_path)
        try: self.process=subprocess.Popen(["afplay",str(path)])
        except Exception as e: messagebox.showerror("Playback unavailable",str(e))

    def stop(self):
        if self.process and self.process.poll() is None: self.process.terminate()

    def save(self):
        r=self.row(); record={"review_id":r.review_id,"relative_path":r.relative_path,
          **{k:v.get() for k,v in self.vars.items()},"reviewer_comments":self.comment_widget.get("1.0","end").strip(),
          "manual_event_onsets_s":self.onsets_widget.get("1.0","end").strip(),"manual_syllable_count":self.manual_count.get(),
          "reviewer_id":self.reviewer.get(),"reviewed_at":datetime.now().astimezone().isoformat(timespec="seconds")}
        self.annotations=self.annotations[~((self.annotations.review_id.astype(str)==str(r.review_id))&(self.annotations.relative_path==r.relative_path))]
        self.annotations=pd.concat([self.annotations,pd.DataFrame([record])],ignore_index=True)
        self.annotations.to_csv(self.annotation_path,index=False)
        refresh_review_summary(self.experiment_root)
        messagebox.showinfo("Saved","Manual annotation saved separately from source data and automated QC.")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--experiment-root",type=Path,default=Path("results/italian_research_phase2")); a=p.parse_args()
    root=tk.Tk(); ReviewApp(root,a.dataset_root,a.experiment_root); root.mainloop()

if __name__=="__main__": main()
