# SPDX-License-Identifier: Apache-2.0
"""Independent acceptance of measured upright pick/place; no simulator imports."""
import argparse
import itertools
import json
import math
from pathlib import Path


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def rotate(q, v):
    w, x, y, z = q
    norm = math.sqrt(sum(t*t for t in q))
    if norm < 1e-8 or not math.isfinite(norm):
        raise ValueError('Invalid object quaternion')
    w, x, y, z = [t/norm for t in q]
    return [
        (1-2*(y*y+z*z))*v[0]+2*(x*y-z*w)*v[1]+2*(x*z+y*w)*v[2],
        2*(x*y+z*w)*v[0]+(1-2*(x*x+z*z))*v[1]+2*(y*z-x*w)*v[2],
        2*(x*z-y*w)*v[0]+2*(y*z+x*w)*v[1]+(1-2*(x*x+y*y))*v[2],
    ]


def validate(r):
    samples, contacts = r['samples'], r['contacts']
    if not samples:
        raise ValueError('Missing measured samples')
    def phase(name):
        rows = [s for s in samples if s['phase'] == name]
        if not rows:
            raise ValueError('Missing phase: '+name)
        return rows
    hold, settle = phase('hold'), phase('settle')
    end = [s for s in settle if s['t'] >= settle[-1]['t']-1.0]
    start = phase('approach')[-1]['position']
    place = end[-1]['position']
    lasttime = samples[-1]['t']
    hand = lambda c: any('/G1/' in actor for actor in c['actors'])
    support = lambda c: any(actor == '/World/Tray' for actor in c['actors'])
    lastcontacts = [c for c in contacts if c['t'] > lasttime-1.0]
    holdcontacts = [c for c in contacts if hold[0]['t']-1/r['fps'] <= c['t'] <= hold[-1]['t']]
    lift = min(s['position'][2]-start[2] for s in hold)
    slip = max(distance(s['position'], hold[0]['position']) for s in hold)
    penetration = max([max(0., -c['separation']) for c in contacts]+[0.])
    tilts = [math.degrees(math.acos(max(-1, min(1, rotate(s['quaternion'], r.get('asset_upright_axis_local', [0,0,1]))[2])))) for s in end]
    drift = max(distance(s['position'], place) for s in end)
    angular_drift = max(math.degrees(2*math.acos(min(1, abs(sum(a*b for a,b in zip(s['quaternion'],end[-1]['quaternion'])))))) for s in end)
    xy = math.hypot(place[0]-r['tray_xy'][0], place[1]-r['tray_xy'][1])
    geometry_present = all(k in r for k in ('asset_bounds_local_m','tray_size_m','tray_top_m'))
    contained, supported = False, False
    if geometry_present:
        corners = list(itertools.product(*zip(*r['asset_bounds_local_m'])))
        bounds = []
        for s in end:
            pts = [[a+b for a,b in zip(rotate(s['quaternion'],v),s['position'])] for v in corners]
            bounds.append(([min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]))
        contained = all(all(lo[i] >= r['tray_xy'][i]-r['tray_size_m'][i]/2+.003 and hi[i] <= r['tray_xy'][i]+r['tray_size_m'][i]/2-.003 for i in range(2)) for lo,hi in bounds)
        supported = all(abs(lo[2]-r['tray_top_m']) < .008 for lo,hi in bounds)
    dt = 1/r['fps']
    checks = {
        'lift_over_0.10m': lift > .10,
        'hold_slip_below_0.015m': slip < .015,
        'hand_contact_during_hold': any(hand(c) for c in holdcontacts),
        'released_no_hand_contact': not any(hand(c) for c in lastcontacts),
        'tray_contact_after_release': any(support(c) for c in lastcontacts),
        'final_upright_below_10deg': max(tilts) < 10,
        'measured_support_geometry_present': geometry_present,
        'asset_footprint_inside_tray': contained,
        'asset_bottom_on_tray': supported,
        'settled_drift_below_0.003m': drift < .003,
        'settled_rotation_below_2deg': angular_drift < 2,
        'settle_evidence_at_least_1s': end[-1]['t']-end[0]['t'] >= 1.-dt-1e-6,
        'contact_penetration_below_0.003m': penetration < .003,
        'sample_rate_matches_fps': abs(samples[0]['t']-dt)<1e-6 and all(abs(b['t']-a['t']-dt)<1e-6 for a,b in zip(samples,samples[1:])),
    }
    return {'status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'lift_m':lift,'hold_slip_m':slip,'maximum_contact_penetration_m':penetration,
            'destination_error_m':xy,'final_tilt_deg':tilts[-1],'maximum_final_tilt_deg':max(tilts),
            'settled_drift_m':drift,'settled_rotation_deg':angular_drift,'final_position_m':place,
            'contact_points':len(contacts),'fps':r['fps'],'physics_hz':r['physics_hz'],
            'fixed_base':r['fixed_base'],'scope':'Measured fixed-base G1 upright mug placement on tray; no locomotion or hardware validation.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('trajectory');a=p.parse_args()
    f=Path(a.trajectory)
    try:
        result=validate(json.loads(f.read_text()))
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        result={'status':'failed','reason':str(exc)}
    (f.parent/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
