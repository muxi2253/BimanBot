"""Build a self-contained browser viewer for the public data package."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'recordings/showcase_public/exports/episode_000000'
OUT=ROOT/'recordings/showcase_public/public-data-viewer.html'

def main():
    frames=[]
    for p in sorted(DATA.glob('frame_*_top_rgb.npy')):
        idx=p.stem.split('_')[1]
        rgb=np.load(p); Image.fromarray(rgb).save(DATA/f'frame_{idx}_top_rgb.png')
        depth=np.load(DATA/f'frame_{idx}_top_depth.npy'); valid=(depth>0)&(depth<4)
        norm=np.clip(depth/4,0,1); dm=cv2.applyColorMap((norm*255).astype('uint8'),cv2.COLORMAP_TURBO)[:,:,::-1]; dm[~valid]=[17,25,38]; Image.fromarray(dm).save(DATA/f'frame_{idx}_top_depth.png')
        seg=np.load(DATA/f'frame_{idx}_top_seg.npy'); colors=np.array([[17,25,38],[72,210,196],[245,155,76]],dtype=np.uint8); Image.fromarray(colors[seg%3]).save(DATA/f'frame_{idx}_top_seg.png')
        frames.append(idx)
    info=json.loads((ROOT/'recordings/showcase_public/lerobot/meta/info.json').read_text())
    html=f'''<!doctype html><meta charset="utf-8"><title>Public robot data viewer</title>
<style>body{{font:16px system-ui;margin:0;background:#101a2b;color:#edf3fa}}main{{max-width:1180px;margin:auto;padding:28px}}h1{{margin:0 0 6px}}.muted{{color:#a4b6ce}}nav{{display:flex;gap:8px;margin:22px 0;flex-wrap:wrap}}button{{padding:10px 15px;border:1px solid #49617d;border-radius:7px;background:#1b2d43;color:#edf3fa;cursor:pointer}}button.active{{background:#36c7b2;color:#071421}}section{{display:none}}section.active{{display:block}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0;background:#17263a;padding:10px;border-radius:8px}}img{{width:100%;display:block}}figcaption{{margin-top:7px;color:#a4b6ce}}pre{{background:#17263a;padding:16px;overflow:auto;border-radius:8px}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style>
<main><h1>Public robot data viewer</h1><div class="muted">opensource-candidate · post-settle SKU initial state · public dual-arm model</div>
<nav><button class="active" onclick="show('frames',this)">RGB / depth / segmentation</button><button onclick="show('points',this)">Point clouds</button><button onclick="show('lerobot',this)">LeRobot</button></nav>
<section id="frames" class="active"><input id="slider" type="range" min="0" max="{len(frames)-1}" value="0" oninput="draw(this.value)"><span id="label"></span><div class="grid"><figure><img id="rgb"><figcaption>Top RGB</figcaption></figure><figure><img id="depth"><figcaption>Metric depth, colorized (0–4 m)</figcaption></figure><figure><img id="seg"><figcaption>Instance segmentation</figcaption></figure></div></section>
<section id="points"><p>World-frame point clouds exported at 15 keyframes.</p><pre>exports/episode_000000/frame_XXXXX_pointcloud.xyz</pre><div class="grid">{''.join(f'<figure><a href="exports/episode_000000/frame_{x}_pointcloud.xyz">frame {x} point cloud (.xyz)</a></figure>' for x in frames)}</div></section>
<section id="lerobot"><pre>{json.dumps(info,indent=2)}</pre><p><a href="lerobot/">Open the LeRobot v2.1 dataset directory</a></p></section>
</main><script>const f={json.dumps(frames)};function draw(i){{let x=f[i];document.getElementById('label').textContent='  frame '+x;for(const k of ['rgb','depth','seg'])document.getElementById(k).src='exports/episode_000000/frame_'+x+'_top_'+k+'.png'}}function show(id,b){{document.querySelectorAll('section').forEach(x=>x.classList.remove('active'));document.getElementById(id).classList.add('active');document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active')}}draw(0)</script>'''
    OUT.write_text(html,encoding='utf-8'); print(OUT)
if __name__=='__main__': main()
