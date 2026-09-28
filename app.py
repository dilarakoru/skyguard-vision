import base64
from datetime import datetime,timezone
from pathlib import Path
from fastapi import FastAPI,Request,HTTPException,Query
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from detector import read_image,infer,MAX_BYTES
ROOT=Path(__file__).resolve().parent
app=FastAPI(title='SkyGuard Vision',version='1.0.0')
@app.get('/',response_class=HTMLResponse)
def index():return (ROOT/'templates/index.html').read_text(encoding='utf-8')
@app.get('/health')
def health():
    import os
    p=Path(os.getenv('SKYGUARD_MODEL',str(ROOT/'models/fire-smoke.pt')))
    return {'status':'ok','model_ready':p.is_file(),'device':'cpu'}
@app.post('/api/detect')
async def detect(request:Request,confidence:float=Query(.35,ge=.05,le=.95)):
    chunks=[];size=0
    async for chunk in request.stream():
        size+=len(chunk)
        if size>MAX_BYTES:raise HTTPException(413,'Image exceeds 12 MiB.')
        chunks.append(chunk)
    try:
        image=read_image(b''.join(chunks))
        boxes,jpeg=await run_in_threadpool(infer,image,confidence)
    except FileNotFoundError as exc:raise HTTPException(503,str(exc)) from exc
    except ValueError as exc:raise HTTPException(400,str(exc)) from exc
    return {'detections':boxes,'count':len(boxes),'processed_at':datetime.now(timezone.utc).isoformat(),
            'image':base64.b64encode(jpeg).decode('ascii'),'threshold':confidence}
