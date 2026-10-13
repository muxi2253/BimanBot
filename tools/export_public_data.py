"""Generate public-model trajectory, RGB/depth/segmentation and point clouds."""
from pathlib import Path
import json, sys, argparse
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from public_demo import PublicPickPlaceDemo
from camera_config import load_scene_with_cameras, CAMERA_CONFIG, RENDER_W, RENDER_H

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,default=50); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); ep=a.out/'episode_000000'; ep.mkdir(exist_ok=True)
    demo=PublicPickPlaceDemo(); frames=[]; phases=[]
    def cb(m,d,phase):
        # The explicit initial marker is emitted after randomized SKU free-fall.
        if phase == 'initial' and not frames:
            np.save(ep/'initial_qpos.npy', d.qpos.copy())
        if frames or phase == 'initial':
            frames.append(d.qpos.copy()); phases.append(phase)
    result=demo.run(seed=a.seed,randomize=True,callback=cb)
    q=np.asarray(frames); np.save(ep/'trajectory_qpos.npy',q); (ep/'trajectory_meta.json').write_text(json.dumps({'record_fps':20,'n_frames':len(q),'nq':int(q.shape[1]),'source':'public_pick_place','initial_state':'post_settle'}))
    (ep/'annotation.json').write_text(json.dumps({'success':True,'seed':a.seed,'source':'opensource-candidate','robot_model':'robot/dual_arm_robot.xml','items':list(result.items),'phases':list(result.phases)}))
    model,data=load_scene_with_cameras(CAMERA_CONFIG); assert model.nq==q.shape[1]; renderer=mujoco.Renderer(model,RENDER_H,RENDER_W)
    try:
      for idx in np.unique(np.linspace(0,len(q)-1,8).astype(int)):
        data.qpos[:]=q[idx]; mujoco.mj_forward(model,data)
        for cam in CAMERA_CONFIG:
          renderer.update_scene(data,camera=cam); rgb=renderer.render().copy();
          renderer.enable_depth_rendering(); renderer.update_scene(data,camera=cam); depth=renderer.render().copy(); renderer.disable_depth_rendering()
          renderer.enable_segmentation_rendering(); renderer.update_scene(data,camera=cam); raw=renderer.render().copy(); renderer.disable_segmentation_rendering()
          np.save(ep/f'frame_{idx:05d}_{cam}_depth.npy',depth); np.save(ep/f'frame_{idx:05d}_{cam}_rgb.npy',rgb)
          geom,kind=raw[:,:,0],raw[:,:,1]; seg=np.zeros((RENDER_H,RENDER_W),np.int32); good=(geom>=0)&(kind==int(mujoco.mjtObj.mjOBJ_GEOM)); safe=np.where(good,geom,0); bodies=model.geom_bodyid[safe]
          for label,name in enumerate(result.items,1):
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name); seg[bodies==bid]=label
          np.save(ep/f'frame_{idx:05d}_{cam}_seg.npy',seg)
          if cam=='top':
            cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,cam); f=RENDER_H/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2)); yy,xx=np.indices(depth.shape); z=depth; valid=(z>0)&(z<4); x=(xx-RENDER_W/2)/f*z; y=(RENDER_H/2-yy)/f*z; camid=data.cam_xpos[cid]; xmat=data.cam_xmat[cid].reshape(3,3); pts=(np.stack((x,y,-z),-1)@xmat.T+camid)[valid]; np.savetxt(ep/f'frame_{idx:05d}_pointcloud.xyz',pts[::2],fmt='%.6f')
    finally: renderer.close()
    print('wrote public episode',ep,'frames',len(q))
if __name__=='__main__': main()
