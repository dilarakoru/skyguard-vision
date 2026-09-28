"""CPU-first image inference, shared by CLI and API."""
from pathlib import Path
import io
import os
import threading
from functools import lru_cache
from PIL import Image, ImageOps, UnidentifiedImageError
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.runtime/yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
MAX_BYTES=12*1024*1024
MAX_PIXELS=20_000_000
Image.MAX_IMAGE_PIXELS=MAX_PIXELS
LOCK=threading.Lock()

def read_image(data):
    if not data or len(data)>MAX_BYTES:raise ValueError('Choose an image smaller than 12 MiB.')
    try:
        image=Image.open(io.BytesIO(data))
        if image.width*image.height>MAX_PIXELS:raise ValueError('Image exceeds 20 megapixels.')
        if image.format not in ('JPEG','PNG','WEBP'):raise ValueError('Supported formats: JPEG, PNG, WebP.')
        image=ImageOps.exif_transpose(image).convert('RGB');image.load();return image
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise ValueError('The file is not a supported readable image.') from exc

@lru_cache(maxsize=2)
def load_model(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError('Model missing. Place your trained weights in models/fire-smoke.pt or set SKYGUARD_MODEL.')
    from ultralytics import YOLO
    model=YOLO(str(p))
    names={str(v).lower() for v in model.names.values()}
    if not {'fire','smoke'}.issubset(names):raise ValueError('Expected a model containing fire and smoke classes.')
    return model

def infer(image,confidence=.35,model_path=None):
    if not .05<=confidence<=.95:raise ValueError('Confidence threshold must be between 0.05 and 0.95.')
    path=str(model_path or os.getenv('SKYGUARD_MODEL',str(ROOT/'models/fire-smoke.pt')))
    model=load_model(path)
    with LOCK: result=model.predict(image,device='cpu',conf=confidence,imgsz=640,save=False,verbose=False)[0]
    boxes=[]
    for box in result.boxes:
        idx=int(box.cls.item())
        boxes.append({'label':model.names[idx],'confidence':round(float(box.conf.item()),4),'xyxy':[round(float(x),2) for x in box.xyxy[0].tolist()]})
    # Ultralytics plots BGR; convert to RGB for PIL/browser output.
    annotated=Image.fromarray(result.plot()[:,:,::-1]);buffer=io.BytesIO();annotated.save(buffer,format='JPEG',quality=90)
    return boxes,buffer.getvalue()
