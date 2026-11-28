# SPDX-License-Identifier: Apache-2.0
import pytest
from rfw_w.scenes.assets.background_tools import screen_floor,bounds,intersects
from rfw_w.scenes.assets.room_tools import RoomAssetError


def plan():
    return {'rooms':[{'id':'study','bounds_xy_m':[0,0,4,4]}], 'corridor':{'x_span_m':[.5,3.5],'y_span_m':[1.5,2.5]}}

def item(pose=(1,3.5,0)):
    return {'role_id':'chair','room_id':'study','dimensions_m':[.5,.5,1.], 'pose_xyz_m':list(pose),'priority':'optional','support_surface':None}

def test_floor_accepts_free_space():
    screen_floor(item(),plan(),{'instances':[]},[])

@pytest.mark.parametrize('pose,code', [((5,3,0),'FURNISHING_OUT_OF_BOUNDS'),((2,2,0),'CORE_KEEP_OUT_VIOLATION'),((1,3.5,1),'FURNISHING_OUT_OF_BOUNDS')])
def test_bad_floor_placements_fail(pose,code):
    with pytest.raises(RoomAssetError,match=code):screen_floor(item(pose),plan(),{'instances':[]},[])

def test_overlap_rejected():
    with pytest.raises(RoomAssetError,match='FURNISHING_COLLISION'):
        screen_floor(item(),plan(),{'instances':[]},[item()])

def test_task_approach_region_reserved():
    core=item((1,.8,0));core.update(priority='core',support_surface={'plane_z_m':1})
    with pytest.raises(RoomAssetError,match='CORE_KEEP_OUT_VIOLATION'):
        screen_floor(item((1,1.2,0)),plan(),{'instances':[core]},[])

def test_rotated_bounds_keep_finite_geometry():
 import math
 value=item();value['dimensions_m']=[1,.2,.5];value['yaw_rad']=math.pi/2
 b=bounds(value);assert b[3]-b[0]==pytest.approx(.2)
 assert b[4]-b[1]==pytest.approx(1)
