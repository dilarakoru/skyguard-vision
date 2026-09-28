import argparse,json
from pathlib import Path
from detector import read_image,infer
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Run local fire/smoke detection on one image.')
    p.add_argument('image',type=Path);p.add_argument('--output',type=Path,default=Path('runs/prediction.jpg'))
    p.add_argument('--confidence',type=float,default=.35);p.add_argument('--model',type=Path)
    a=p.parse_args();boxes,jpeg=infer(read_image(a.image.read_bytes()),a.confidence,a.model)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(jpeg)
    a.output.with_suffix('.json').write_text(json.dumps({'detections':boxes,'threshold':a.confidence},indent=2),encoding='utf-8')
    print(f'{len(boxes)} detections; saved {a.output}')
