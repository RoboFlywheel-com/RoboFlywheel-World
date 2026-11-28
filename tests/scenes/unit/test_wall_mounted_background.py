# SPDX-License-Identifier: Apache-2.0
import pytest
from rfw_w.scenes.assets.background_tools import screen_wall
from rfw_w.scenes.assets.room_tools import RoomAssetError

def anchor():return {'room_id':'study','normal':[0,-1,0],'axis':'y','plane_m':3.95,'span_m':[0,4],'height_m':[0,2.8]}
def item():return {'role_id':'mirror','room_id':'study','dimensions_m':[.5,.05,.7], 'pose_xyz_m':[1,3.923,1.4], 'mount_normal':[0,-1,0]}
def plan():return {'corridor':{'x_span_m':[.5,3.5],'y_span_m':[1.5,2.5]}}
def test_wall_accepts_measured_face():screen_wall(item(),anchor(),plan(),{'instances':[]},[])
@pytest.mark.parametrize('fields,code',[
    ({'mount_normal':[0,1,0]},'WALL_SIDE_MISMATCH'),
    ({'pose_xyz_m':[1,4,1.4]},'WALL_SIDE_MISMATCH'),
    ({'pose_xyz_m':[.1,3.923,1.4]},'WALL_OPENING_OVERLAP'),
    ({'pose_xyz_m':[1,3.923,2.6]},'WALL_OPENING_OVERLAP'),
    ({'room_id':'kitchen'},'WALL_SIDE_MISMATCH')])
def test_bad_wall_installations_rejected(fields,code):
    value=item();value.update(fields)
    with pytest.raises(RoomAssetError,match=code):screen_wall(value,anchor(),plan(),{'instances':[]},[])
def test_wall_overlap_rejected():
    with pytest.raises(RoomAssetError,match='WALL_FIXTURE_COLLISION'):
        screen_wall(item(),anchor(),plan(),{'instances':[]},[item()])
