# SPDX-License-Identifier: Apache-2.0
"""Restore original textured visual assets in MuJoCo; preserve installation physics."""
import argparse,json,os
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--delivery',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);os.environ.setdefault('MUJOCO_GL','egl')
    from rfw_w.scenes.validation.mujoco_installation_appearance import build
    from rfw_w.scenes.validation.verify_mujoco_installation_appearance import verify
    ledger=build(a.delivery,a.output_dir);verification=verify(a.output_dir)
    (a.output_dir/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    if not verification['verified']:raise ValueError('APPEARANCE_PHYSICS_CHANGED')
    import mujoco
    from PIL import Image
    config=json.loads((a.delivery/'installation.json').read_text());model=mujoco.MjModel.from_xml_path(str(a.output_dir/'appearance.xml'));data=mujoco.MjData(model)
    for n,v in config['reference_joints'].items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n);data.qpos[model.jnt_qposadr[j]]=v
    model.vis.headlight.ambient[:]=[.4,.4,.4];model.vis.headlight.diffuse[:]=[.6,.6,.6]
    model.vis.global_.offwidth=960;model.vis.global_.offheight=720
    mujoco.mj_forward(model,data);renderer=mujoco.Renderer(model,height=720,width=960)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    support=next(i for i in config['instances'] if i['role_id']==config['support_role_id'])
    # Keep far walls visible; hide only left/bottom external faces for this viewing azimuth.
    hidden=[]
    for surface in ledger['surfaces']:
        if surface['surface_id'].startswith(('wall_left:','wall_bottom_')):
            model.geom_group[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,surface['geom'])]=4;hidden.append(surface['surface_id'])
    option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[2]=1
    report={'backend':'mujoco','version':mujoco.__version__,'source_scene_sha256':digest(a.delivery/'scene.xml'),
            'appearance':'source visual OBJ/MTL textures and native room material layers','hidden_render_only':hidden,
            'renderer_limits':ledger['renderer_limits'],'robot_pose':'declared reference posture','grasp':'not_run','motion':'not_run','images':{}}
    for name,lookat,distance,azimuth,elevation in [
        ('installation',[support['pose_xyz_m'][0],support['pose_xyz_m'][1],support['support_surface']['plane_z_m']],2.3,120,-35),
        ('room',[-2,2,1],6.5,120,-28)]:
        camera.lookat[:]=lookat;camera.distance=distance;camera.azimuth=azimuth;camera.elevation=elevation
        renderer.update_scene(data,camera=camera,scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE]=0
        Image.fromarray(renderer.render()).save(a.output_dir/(name+'.png'));report['images'][name]=digest(a.output_dir/(name+'.png'))
    renderer.close();(a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
