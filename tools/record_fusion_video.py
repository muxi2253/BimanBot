"""Turn a generated public spatial-fusion run into a presentation MP4."""
from pathlib import Path
import argparse, json
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--src',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    w=imageio.get_writer(a.out,fps=24,codec='libx264',quality=8,macro_block_size=1)
    try:
        for si in range(0,20,2):
            d=a.src/f'sample_{si:05d}'; truth=np.load(d/'current_depth_truth.npy'); rgb=np.load(d/'rgb.npy')
            prior=np.load(d/'prior_depth.npy'); obs=np.load(d/'observation_depth.npy');
            panels=[('RGB',rgb)]
            for name,z in [('PRIOR',prior),('OBSERVATION',obs),('TRUTH',truth)]:
                err=np.clip(np.abs(z-truth)/.1,0,1); x=cv2.applyColorMap((err*255).astype('uint8'),cv2.COLORMAP_TURBO)[:,:,::-1]; x[(z<=0)|(truth<=0)]=[17,25,38]; panels.append((name,x))
            board=Image.new('RGB',(1280,720),'#101a2b'); dr=ImageDraw.Draw(board); dr.text((32,22),'PUBLIC SPATIAL FUSION',font=font,fill='white'); dr.text((32,60),'opensource-candidate | four non-learning baselines | seed 7',font=small,fill='#a4b6ce')
            for j,(name,x) in enumerate(panels):
                xx=32+(j%2)*624; yy=115+(j//2)*285; dr.text((xx,yy),name,font=small,fill='#56ddcb'); board.paste(Image.fromarray(x).resize((576,240)),(xx,yy+28))
            dr.text((32,670),'Error scale: blue 0 mm → red 100+ mm | dark pixels are invalid',font=small,fill='#a4b6ce')
            for _ in range(12): w.append_data(np.asarray(board))
    finally: w.close()
    print(a.out)
if __name__=='__main__': main()
