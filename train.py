"""Explicit training command; no training or downloads during app startup."""
import argparse
from pathlib import Path
from detector import ROOT
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True,help='Local YOLO dataset YAML')
    p.add_argument('--weights',type=Path,required=True,help='Local architecture-compatible starting weights')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=4);a=p.parse_args()
    if not a.data.is_file() or not a.weights.is_file():p.error('Both data YAML and weights must exist locally.')
    from ultralytics import YOLO
    YOLO(str(a.weights)).train(data=str(a.data.resolve()),epochs=a.epochs,device=a.device,batch=a.batch,imgsz=640,project=str(ROOT/'runs'),name='training')
