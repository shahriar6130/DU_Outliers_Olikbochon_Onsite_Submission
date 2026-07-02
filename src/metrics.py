
"""
Metrics Module
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

@dataclass
class AverageMeter:
    val: float=0.0
    avg: float=0.0
    sum: float=0.0
    count: int=0
    def reset(self):
        self.val=self.avg=self.sum=0.0
        self.count=0
    def update(self,value:float,n:int=1):
        self.val=float(value)
        self.sum+=float(value)*n
        self.count+=n
        self.avg=self.sum/self.count if self.count else 0.0

def _np(x:Sequence):
    if hasattr(x,"detach"):
        x=x.detach().cpu().numpy()
    return np.asarray(x)

def compute_metrics(y_true,y_pred)->Dict[str,float]:
    yt=_np(y_true); yp=_np(y_pred)
    return {
        "accuracy": float(accuracy_score(yt,yp)),
        "precision_macro": float(precision_score(yt,yp,average="macro",zero_division=0)),
        "recall_macro": float(recall_score(yt,yp,average="macro",zero_division=0)),
        "macro_f1": float(f1_score(yt,yp,average="macro",zero_division=0)),
        "hallucination_f1": float(f1_score(yt,yp,pos_label=0,average="binary",zero_division=0)),
    }

def build_confusion_matrix(y_true,y_pred):
    return confusion_matrix(_np(y_true),_np(y_pred))

def build_classification_report(y_true,y_pred):
    return classification_report(_np(y_true),_np(y_pred),digits=4,zero_division=0)

def print_metrics(metrics:Dict[str,float]):
    print("="*60)
    for k,v in metrics.items():
        print(f"{k:20}: {v:.4f}")
    print("="*60)
