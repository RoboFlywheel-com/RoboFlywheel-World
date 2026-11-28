#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure room type/scale against inputs. Baseline and builtin runs do not certify Infinigen."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path.cwd()
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.contracts.room_plan import floor_plan_payload
from rfw_w.scenes.contracts.shell_build import plan_gaps
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.room import fit_room_around_cell
from rfw_w.scenes.solvers.room_plan import plan_from_fitted
from rfw_w.scenes.solvers.shell_providers import provider_named, minimum_door_width_m


def measure(case, out, provider, baseline):
    started = time.monotonic()
    out.mkdir()
    result = {'case': case['id'], 'kind': case['kind'], 'scale': case['scale'],
              'seed': case['seed'], 'gaps': [], 'provider': provider}
    try:
        if 'prompt' in case:
            from rfw_w.scenes.agents.scene_compiler import compile_scene
            from rfw_w.scenes.agents.llm_client import LLMConfig
            config = LLMConfig(
                base_url=os.environ.get('PVSCENE_VLM_BASE_URL') or os.environ.get('SIMFOUNDRY_OPENAI_BASE_URL',''),
                model=os.environ.get('PVSCENE_COMPILER_MODEL') or os.environ.get('PVSCENE_VLM_MODEL',''),
                api_key_env=os.environ.get('PVSCENE_VLM_API_KEY_ENV','DASHSCOPE_API_KEY'),
                wire=os.environ.get('PVSCENE_VLM_WIRE','chat_completions'))
            compilation = compile_scene(config,prompt=case['prompt'],seed=case['seed'],
                                        scene_id=case['id'],on_request=lambda:None)
            (out/'compilation.json').write_text(json.dumps(compilation.record,indent=2))
            if compilation.spec is None:
                raise ValueError('ROOM_PROMPT_COMPILATION_FAILED')
            spec=compilation.spec
            if spec.room.size_policy != 'inferred':
                result['gaps'].append('ROOM_PROMPT_SIZE_SOURCE_MISMATCH')
        else:
            raw = json.loads((ROOT / case['spec']).read_text())
            raw['room'] = {'size_bounds_m': case['bounds']}
            if not baseline:
                raw['room'].update(room_kind=case['kind'], scale_class=case['scale'], size_policy='explicit')
            spec = SceneSpec.from_dict(raw)
        (out / 'effective_spec.json').write_text(json.dumps(spec.to_dict(), indent=2))
        # Freeze the same task core across room seeds/scales, as in the production boundary.
        core = solve_cell(spec, seed=spec.seed)
        before = core.core_hash
        fitted = fit_room_around_cell(spec, core, room_seed=case['seed'])
        plan = plan_from_fitted(spec, core, fitted, room_seed=case['seed'])
        (out / 'plan.json').write_text(json.dumps(plan.to_dict(), indent=2))
        result['core_hash'] = before
        result['room_fit'] = dict(opening_attempts=fitted.opening_attempts, shape_fallback=fitted.shape_fallback, size_margin_m=fitted.size_margin_m)
        result['actual_kind'] = plan.rooms[0].kind
        result['floor_plan_keys'] = list(floor_plan_payload(plan)['rooms'])
        polygon = Polygon(plan.rooms[0].polygon_xy)
        x0,y0,x1,y1 = polygon.bounds()
        result['spans_m'] = [x1-x0,y1-y0,plan.rooms[0].ceiling_height_m]
        result['area_m2'] = polygon.area
        if plan.rooms[0].kind != case['kind']:
            result['gaps'].append('ROOM_KIND_MISMATCH')
        if before != core.core_hash or before != plan.core.core_hash:
            result['gaps'].append('ROOM_CORE_MUTATED')
        if any(not lo-1e-4 <= v <= hi+1e-4 for v,(lo,hi) in zip(result['spans_m'],case['bounds'])):
            result['gaps'].append('ROOM_SIZE_OUTSIDE_INPUT')
        build = provider_named(provider).build(plan, output_dir=out/'build')
        (out/'shell_build.json').write_text(json.dumps(build.to_dict(),indent=2))
        result['gaps'].extend(str(g) for g in plan_gaps(plan,build,minimum_door_width_m=minimum_door_width_m()))
        result['build_source'] = build.shell.source
        if provider == 'infinigen':
            result['planned_area_m2']=result['area_m2']
            result['area_m2']=build.measurements.get('room.floor_area_m2',0)
            if result['area_m2'] <= 0: result['gaps'].append('ROOM_FLOOR_AREA_NOT_MEASURED')
            bbox=build.measurements['room.shell_bbox_m']
            result['built_spans_m']=[hi-lo for hi,lo in zip(bbox['max'],bbox['min'])]
            if any(not lo-1e-3 <= v <= hi+1e-3 for v,(lo,hi) in zip(result['built_spans_m'],case['bounds'])):
                result['gaps'].append('ROOM_BUILT_SIZE_OUTSIDE_INPUT')
            source=Path(build.geometry_uris[0])
            state=json.loads((source.parent/'solve_state.json').read_text())
            rooms={k:v for k,v in state['objs'].items() if 'Semantics(room)' in v.get('tags',[])}
            result['built_room_tags']={k:v.get('tags',[]) for k,v in rooms.items()}
            if not any(f'Semantics({case["kind"]})' in v.get('tags',[]) for v in rooms.values()):
                result['gaps'].append('ROOM_BUILT_KIND_MISMATCH')
    except Exception as error:
        result['gaps'].append(f'{type(error).__name__}({error})')
    result['passed'] = not result['gaps']
    result['elapsed_s'] = round(time.monotonic()-started,3)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--provider',choices=['builtin','infinigen'],default='infinigen')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--mode',choices=['json','prompt','all'],default='all')
    parser.add_argument('--jobs',type=int,choices=range(1,5),default=1)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): parser.error('use a fresh output directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (ROOT/'.env.local').is_file():
        for line in (ROOT/'.env.local').read_text().splitlines():
            line=line.strip()
            if line and not line.startswith('#') and '=' in line:
                key,_,value=line.partition('=');os.environ.setdefault(key.strip(),value.strip().strip('\'"'))
    matrix=json.loads(args.matrix.read_text())
    (args.output_dir/'matrix.json').write_text(json.dumps(matrix,indent=2))
    cases=[]
    if args.baseline or args.mode in ('json','all'): cases.extend(matrix['cases'])
    if not args.baseline and args.mode in ('prompt','all'): cases.extend(matrix.get('prompt_cases',[]))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows=list(pool.map(lambda c: measure(c,args.output_dir/c['id'],args.provider,args.baseline),cases))
    gaps=[]
    if not args.baseline and args.mode in ('json','all'):
        for kind in ('office','kitchen','bedroom'):
            group=[r for r in rows if r['kind']==kind and not r['case'].startswith('prompt_')]
            hashes={r.get('core_hash') for r in group}
            if len(hashes)!=1: gaps.append(f'ROOM_CORE_CHANGED_ACROSS_SCALE({kind})')
            for seed in (20,21):
                by_scale={r['scale']:r for r in group if r['seed']==seed}
                if (by_scale.get('large',{}).get('area_m2',0)
                        <= by_scale.get('small',{}).get('area_m2',float('inf'))):
                    gaps.append(f'ROOM_LARGE_NOT_LARGER({kind},{seed})')
    negatives=[]
    if not args.baseline:
        from dataclasses import replace
        from rfw_w.scenes.contracts.scene_spec import RoomSpec
        reference=SceneSpec.from_dict(json.loads((ROOT/matrix['cases'][0]['spec']).read_text()))
        for name,expected in [('invalid_kind','ROOM_KIND_UNSUPPORTED'),('tiny_explicit','ROOM_SIZE_CONFLICT')]:
            observed=''
            try:
                if name=='invalid_kind':RoomSpec.from_dict({'room_kind':'unsupported','size_bounds_m':[[4,5],[3,4],[2.7,3]]})
                else:
                    core=solve_cell(reference,seed=reference.seed)
                    tiny=RoomSpec.from_dict({'room_kind':'office','size_bounds_m':[[.5,.6],[.5,.6],[2.7,3]]})
                    fit_room_around_cell(replace(reference,room=tiny),core,room_seed=20)
            except ValueError as error:observed=str(error)
            negatives.append({'name':name,'expected':expected,'observed':observed,'passed':expected in observed})
        if not all(n['passed'] for n in negatives):gaps.append('ROOM_NEGATIVE_DID_NOT_FAIL_AS_EXPECTED')
    passed=bool(rows) and all(r['passed'] for r in rows) and not gaps
    report={'scope':args.mode,'baseline':args.baseline,'provider':args.provider,'cases':rows,
            'gaps':gaps,'negative_cases':negatives,'passed':passed,
            'spec_accepted':passed and not args.baseline and args.mode=='all'
                            and args.provider=='infinigen' and len(rows)==21}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':report['passed'],'cases':len(rows),'failures':sum(not r['passed'] for r in rows)}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
