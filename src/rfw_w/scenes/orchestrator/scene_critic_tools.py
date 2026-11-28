# SPDX-License-Identifier: Apache-2.0
"""Generic read-only profile dispatch; explicit external model and robot executors."""
import json
from pathlib import Path
import time
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_critique import (
    SCHEMA,require,validate_request,validate_visual_response,issue,
)
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
from rfw_w.scenes.orchestrator.layout_revision_tools import read,write,verify_layout_revision
from rfw_w.scenes.orchestrator.rigid_asset_tools import _verify_bundle
from rfw_w.scenes.geometry.layout_revision import measure_layout


def support_margin(core,role_id):
    from rfw_w.scenes.geometry.layout_revision import footprint
    item=next(i for i in core['instances'] if i['role_id']==role_id)
    if not item.get('support_role_id'):return None
    support=next(i for i in core['instances'] if i['role_id']==item['support_role_id'])['support_surface']
    xs=[p[0] for p in support['polygon_xy']];ys=[p[1] for p in support['polygon_xy']]
    box=footprint(item).bounds()
    return min(box[0]-min(xs),box[1]-min(ys),max(xs)-box[2],max(ys)-box[3])


def margin_issues(graph,core,scope_ids):
    nodes={n['id']:n for n in graph.to_dict()['nodes']}
    result=[]
    for item in core['instances']:
        if item['role_id'] not in nodes or not (item['role_id'] in scope_ids or item['room_id'] in scope_ids):continue
        margin=support_margin(core,item['role_id'])
        if margin is not None and margin<.06:
            result.append(issue('static',[item['role_id']],'SUPPORT_EDGE_MARGIN',
                'measured support edge margin '+format(margin,'.6f')+' m below 0.060000 m soft review threshold',
                severity='warning',operation='translate'))
    return result


class ModelBudget:
    def __init__(self,limit):
        require(type(limit) is int and limit>=0,'BUDGET_EXHAUSTED','model limit')
        self.limit=limit;self.used=0
    def spend(self):
        require(self.used<self.limit,'BUDGET_EXHAUSTED','model requests')
        self.used+=1


def critique_scene(source,output,*,scope_ids,profiles=('static',),expected_version,
                   robot_delivery=None,robot_executor=None,render_manifest=None,config=None,budget=None,cache=None):
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();graph=read_graph(source)
    require(graph.content_hash==expected_version,'CRITIC_INPUT_INVALID','version')
    validate_request(graph,scope_ids,profiles);source_hashes=artifact_digest(source)
    cache_key=content_hash({'scene':source_hashes,'scope':list(scope_ids),'profiles':list(profiles),
        'robot':artifact_digest(robot_delivery) if robot_delivery else None,
        'render':artifact_digest(render_manifest.parent) if render_manifest else None,
        'model':{'model':config.model,'wire':config.wire,'base_url':config.base_url} if config else None,
        'critic_version':'pvscene.scene-critique/v1','edge_margin_m':.06})
    if cache is not None and cache_key in cache:
        prior=cache[cache_key]
        require(content_hash({k:v for k,v in prior['report'].items() if k!='content_hash'})==prior['report']['content_hash'],
                'CRITIC_INPUT_INVALID','cached verdict')
        if prior.get('render_inputs'):
            import shutil
            shutil.copytree(prior['render_inputs'],output/'render_inputs')
            write(output,'render_inputs.json',prior['render_manifest'])
        from copy import deepcopy
        report=deepcopy(prior['report']);report['model_requests']=0;report['cache_hit']=True
        report['elapsed_s']=round(time.monotonic()-started,3)
        report.pop('content_hash');report['content_hash']=content_hash(report);write(output,'critique.json',report)
        require(artifact_digest(source)==source_hashes,'CRITIC_SOURCE_CHANGED','cached source')
        return report
    inputs={'scene':source_hashes};results={};issues=[];calls_before=budget.used if budget else 0
    for profile in profiles:
        if profile=='static':
            try:
                if (source/'layout_revision.json').exists():verify_layout_revision(source)
                _,core,checked=_verify_bundle(source)
                geometry=measure_layout(core,read(source,'background_ledger.json'),read(source,'geometry/house_plan.json'))
                results[profile]={'status':'passed','hard_pass':True,'measurement':geometry,'assembly':checked}
                local_issues=margin_issues(graph,core,scope_ids);issues.extend(local_issues)
                results[profile]['quality_pass']=not local_issues
            except ValueError as error:
                code=getattr(error,'code','CRITIC_INPUT_INVALID')
                results[profile]={'status':'passed','hard_pass':False,'failure_code':code}
                issues.append(issue('static',scope_ids,code,'independent current geometry rejected input'))
        elif profile=='robot_layout':
            if robot_delivery is None or robot_executor is None:
                results[profile]={'status':'not_run','reason':'current robot delivery/executor missing'};continue
            robot_hashes=artifact_digest(robot_delivery);inputs['robot_layout']=robot_hashes
            require(read(robot_delivery,'source_graph.json')['content_hash']==graph.content_hash and
                    read(robot_delivery,'source_artifacts.json')==source_hashes,
                    'CRITIC_INPUT_INVALID','robot evidence version/assets')
            measured=robot_executor(robot_delivery)
            results[profile]={'status':'passed','hard_pass':measured.get('verified') is True,'measurement':measured}
            if not measured.get('verified'):
                issues.append(issue('robot_layout',scope_ids,measured.get('failure_code','ROBOT_LAYOUT_INPUT_INVALID'),
                                    'independent robot layout rejected input',operation='revise_robot_installation'))
            require(artifact_digest(robot_delivery)==robot_hashes,'CRITIC_SOURCE_CHANGED','robot evidence')
        else:
            require(render_manifest is not None,'CRITIC_STALE_RENDER','current render missing')
            manifest=read(render_manifest.parent,render_manifest.name)
            require(set(manifest)=={'graph_hash','scene_digest','images','content_hash'} and
                    content_hash({k:v for k,v in manifest.items() if k!='content_hash'})==manifest['content_hash'] and
                    manifest['graph_hash']==graph.content_hash and manifest['scene_digest']==content_hash(source_hashes),
                    'CRITIC_STALE_RENDER','render dependencies')
            require(isinstance(manifest['images'],dict) and bool(manifest['images']),'CRITIC_STALE_RENDER','images')
            from rfw_w.scenes.geometry.surface_appearance import checked_path,digest
            from rfw_w.scenes.agents.llm_client import call_vision,image_data_url,require_external_permission
            images=[]
            for ref,expected in manifest['images'].items():
                path=checked_path(render_manifest.parent,ref)
                require(digest(path)==expected,'CRITIC_STALE_RENDER','image hash');images.append(image_data_url(str(path)))
            import shutil
            render_copy=output/'render_inputs';render_copy.mkdir()
            for index,(ref,expected) in enumerate(manifest['images'].items()):
                shutil.copyfile(checked_path(render_manifest.parent,ref),render_copy/(str(index)+'.png'))
            write(output,'render_inputs.json',{'images':{str(index)+'.png':expected for index,expected in enumerate(manifest['images'].values())},
                                              'manifest':manifest})
            inputs['visual']={'manifest':manifest,'manifest_ref':render_manifest.name}
            require_external_permission()
            require(config is not None and budget is not None,'BUDGET_EXHAUSTED','explicit model/budget missing')
            semantics={'graph_hash':graph.content_hash,'scope_ids':list(scope_ids),
                'roles':[{'id':n['id'],'category':n['category'],'room_id':n['room_id']}
                         for n in graph.to_dict()['nodes'] if n['kind']=='asset_role'],
                'task':{'initial_state':graph.to_dict()['initial_state'],'goals':graph.to_dict()['goals']}}
            budget.spend()
            response=call_vision(config,system_prompt='Review current scene images against task semantics. Return only JSON with graph_hash, quality_pass (boolean), issues (list). Each issue has scope_ids (known IDs), severity (warning or error), code (VISUAL_QUALITY only), detail, suggested_operation (translate, rotate, move_component, revise_robot_installation or inspect). Be specific. Do not claim physics, IK or task success. Never mutate objects.',
                                 user_prompt=json.dumps(semantics,ensure_ascii=False),images=images)
            raw=json.loads(response.text.strip().removeprefix('```json').removesuffix('```').strip())
            parsed=validate_visual_response(graph,raw,scope_ids);issues.extend(parsed)
            results[profile]={'status':'passed','quality_pass':raw['quality_pass'],'model_id':config.model,
                             'model_requests':1,'response':raw}
    require(artifact_digest(source)==source_hashes,'CRITIC_SOURCE_CHANGED','scene')
    hard=[r.get('hard_pass') for r in results.values() if 'hard_pass' in r]
    report={'schema_version':SCHEMA,'status':'passed','graph_hash':graph.content_hash,
            'scope_ids':list(scope_ids),'profiles':list(profiles),'profile_results':results,'issues':issues,
            'hard_pass':all(hard) if hard else None,
            'quality_pass':all(r['quality_pass'] for r in results.values() if 'quality_pass' in r)
                           if any('quality_pass' in r for r in results.values()) else None,
            'source_unchanged':True,'input_hashes':inputs,'model_requests':(budget.used-calls_before) if budget else 0,
            'elapsed_s':round(time.monotonic()-started,3)}
    report['content_hash']=content_hash(report);write(output,'critique.json',report)
    if cache is not None:
        from copy import deepcopy
        cache[cache_key]={'report':deepcopy(report),'render_inputs':output/'render_inputs' if (output/'render_inputs').exists() else None,
                          'render_manifest':read(output,'render_inputs.json') if (output/'render_inputs.json').exists() else None}
    return report


def verify_critique(source,delivery,*,robot_delivery=None,robot_executor=None):
    report=read(delivery,'critique.json');graph=read_graph(source)
    require(content_hash({k:v for k,v in report.items() if k!='content_hash'})==report['content_hash'] and
            report['graph_hash']==graph.content_hash and artifact_digest(source)==report['input_hashes']['scene'],
            'CRITIC_SOURCE_CHANGED','critique dependencies')
    validate_request(graph,report['scope_ids'],report['profiles'])
    # Replay local profile measurements without contacting a model or writing into the source.
    if 'static' in report['profiles']:
        expected=report['profile_results']['static']['hard_pass']
        try:
            if (source/'layout_revision.json').exists():verify_layout_revision(source)
            _,core,_=_verify_bundle(source);measure_layout(core,read(source,'background_ledger.json'),read(source,'geometry/house_plan.json'))
            actual=True
            local=margin_issues(graph,core,report['scope_ids'])
            require([i for i in report['issues'] if i['source']=='static']==local,
                    'CRITIC_INPUT_INVALID','static issue replay')
        except ValueError:actual=False
        require(expected==actual,'CRITIC_INPUT_INVALID','static replay')
    if 'visual' in report['profiles']:
        from rfw_w.scenes.geometry.surface_appearance import digest,checked_path
        saved=read(delivery,'render_inputs.json')
        require(saved['manifest']==report['input_hashes']['visual']['manifest'],
                'CRITIC_STALE_RENDER','saved render manifest')
        for ref,expected in saved['images'].items():
            require(digest(checked_path(delivery/'render_inputs',ref))==expected,
                    'CRITIC_STALE_RENDER','saved render image')
        parsed=validate_visual_response(graph,report['profile_results']['visual']['response'],report['scope_ids'])
        require(parsed==[i for i in report['issues'] if i['source']=='visual'],
                'CRITIC_RESPONSE_INVALID','visual issues replay')
        require(report['profile_results']['visual']['quality_pass']==
                report['profile_results']['visual']['response']['quality_pass'],
                'CRITIC_RESPONSE_INVALID','visual quality replay')
    if 'robot_layout' in report['profiles'] and report['profile_results']['robot_layout']['status']=='passed':
        require(robot_delivery is not None and robot_executor is not None,'CRITIC_INPUT_INVALID','robot replay missing')
        require(artifact_digest(robot_delivery)==report['input_hashes']['robot_layout'],'CRITIC_SOURCE_CHANGED','robot replay')
        actual=robot_executor(robot_delivery)
        require(actual==report['profile_results']['robot_layout']['measurement'],'CRITIC_INPUT_INVALID','robot replay')
    hard=[r['hard_pass'] for r in report['profile_results'].values() if 'hard_pass' in r]
    quality=[r['quality_pass'] for r in report['profile_results'].values() if 'quality_pass' in r]
    require(report['hard_pass']==(all(hard) if hard else None) and
            report['quality_pass']==(all(quality) if quality else None),
            'CRITIC_RESPONSE_INVALID','aggregate verdict')
    return {'verified':True,'problems':[],'source_unchanged':True,'model_requests':0,
            'visual_model_replay':'not_run','hard_pass':report['hard_pass'],'quality_pass':report['quality_pass']}
