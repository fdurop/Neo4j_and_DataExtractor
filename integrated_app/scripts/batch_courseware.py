"""Resumable, sequential, bounded courseware batch via the real local web API.

Source files are read-only. STOP stops new work and cancels the active web job.
Every segment records the original file hash and inclusive original page range.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'engines/extraction'))
BASE='http://127.0.0.1:8769'

def now():return datetime.now(timezone.utc).isoformat()
def read(path):
    # Windows may briefly deny a read while the worker replaces its status file.
    for attempt in range(21):
        try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except (PermissionError, json.JSONDecodeError):
            if attempt==20:raise
            time.sleep(0.25)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)
def session(path):
    credentials=read(path);s=requests.Session();s.cookies.update(credentials['cookies'])
    r=s.get(BASE+'/api/session',timeout=10);r.raise_for_status()
    if not r.json().get('authenticated'):
        r=s.post(BASE+'/login',json={'username':credentials['username'],'password':credentials['password']},timeout=15);r.raise_for_status()
    return s
def post(s,path,**kwargs):
    r=s.post(BASE+path,timeout=(10,60),**kwargs)
    if r.status_code>=300:raise RuntimeError(f'API {path}: HTTP {r.status_code}')
    return r.json()

def prepare(args):
    import fitz
    from validate_full_document_api import prepare_document
    destination=args.state.parent;destination.mkdir(parents=True,exist_ok=True)
    if args.state.exists():
        state=read(args.state)
        if state.get('status')!='preparing':raise RuntimeError('Existing prepared batch: use run/status')
    else:
        s=session(args.session)
        course_name='机电系统原理及实践·批量测试 2026-11-01'
        courses=s.get(BASE+'/api/courses',timeout=15).json()['courses']
        course=next((c for c in courses if c['name']==course_name),None) or post(s,'/api/courses',json={'name':course_name})
        state={'created_at':now(),'status':'preparing','course':course,'sources':[],'units':[],
               'cost_limit_yuan':30,'budget_note':'Conservative text/vision ledger estimate; embedding and storage costs are not included.'}
        write(args.state,state)
    completed_sha=None
    prior=ROOT/'runtime/fullflow_test_20261101.json'
    if prior.exists():
        p=read(prior);job=read(ROOT/'runtime/jobs'/p['job_id']/'job.json')
        if job['status']=='succeeded':completed_sha=job['source_sha256']
    known={x['source_sha256'] for x in state['sources']}
    seen=set(known)
    for source in sorted(args.folder.rglob('*')):
        if source.suffix.lower() not in {'.pdf','.pptx','.ppt'}:continue
        source_hash=sha(source)
        if source_hash in known:continue
        entry={'original_file':str(source),'source_sha256':source_hash,'name':source.name,'unit_ids':[]}
        if source_hash==completed_sha:
            entry.update(status='already_completed',course_id=p['course']['id'],job_id=p['job_id'])
            state['sources'].append(entry);seen.add(source_hash);write(args.state,state);continue
        if source_hash in seen:
            entry['status']='duplicate';state['sources'].append(entry);write(args.state,state);continue
        pdf=source
        if source.suffix.lower()!='.pdf':
            pdf=destination/'converted'/(source.stem+'.pdf')
            if not pdf.exists():
                entry['status']='needs_pdf_conversion';state['sources'].append(entry);write(args.state,state);continue
        with fitz.open(pdf) as document:
            entry['page_count']=len(document);entry['converted_pdf']=str(pdf) if pdf!=source else None
            def add_part(start,end):
                identity='unit_'+hashlib.sha256((source_hash+f':{start}:{end}').encode()).hexdigest()[:24]
                existing=next((x for x in state['units'] if x['id']==identity),None)
                if existing:entry['unit_ids'].append(identity);return
                filename=f'{source.stem}__p{start+1:03d}-{end:03d}.pdf'
                path=destination/'inputs'/source_hash[:12]/filename;path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():
                    part=fitz.open();part.insert_pdf(document,from_page=start,to_page=end-1)
                    part.set_metadata({'title':source.stem+f' (original pages {start+1}-{end})','subject':f'Original SHA256 {source_hash}'})
                    part.save(path,garbage=4,deflate=True);part.close()
                preflight=destination/'preflight'/identity
                manifest=prepare_document(path,preflight,course_name=state['course']['name'],course_id=state['course']['id'],job_id='job_'+hashlib.sha256(identity.encode()).hexdigest()[:32])
                if manifest.get('preparation_errors'):raise RuntimeError('PDF preparation failed: '+filename)
                count=len(manifest['tasks'])
                if count>50 and end-start>1:
                    middle=(start+end)//2;add_part(start,middle);add_part(middle,end);return
                if count>65:raise RuntimeError('Single page exceeds safe visual-task limit: '+filename)
                unit={'id':identity,'input_path':str(path),'input_sha256':sha(path),'filename':filename,
                      'source_file':source.name,'source_sha256':source_hash,'original_page_start':start+1,'original_page_end':end,
                      'page_count':end-start,'visual_task_count':count,'status':'pending','attempts':0}
                state['units'].append(unit);entry['unit_ids'].append(identity);write(args.state,state)
                print(json.dumps({'prepared':filename,'pages':end-start,'visual_tasks':count},ensure_ascii=False),flush=True)
            for start in range(0,len(document),12):add_part(start,min(start+12,len(document)))
        entry['status']='prepared';state['sources'].append(entry);seen.add(source_hash);write(args.state,state)
    state.update(status='prepared',prepared_at=now());write(args.state,state)
    print(json.dumps({'prepared_units':len(state['units']),'source_files':len(state['sources']),'course':state['course']},ensure_ascii=False),flush=True)

def job_for(unit,course_id):
    if unit.get('job_id'):
        path=ROOT/'runtime/jobs'/unit['job_id']/'job.json'
        return read(path) if path.exists() else None
    matches=[]
    for path in (ROOT/'runtime/jobs').glob('job_*/job.json'):
        value=read(path)
        if value['course_id']==course_id and value.get('source_sha256')==unit['input_sha256']:
            matches.append(value)
    if matches:
        matches.sort(key=lambda j:j['created_at']);return matches[-1]
    return None

def spent(state):
    total=0
    for unit in state['units']:
        if unit.get('job_id'):
            path=ROOT/'runtime/jobs'/unit['job_id']/'api_budget.json'
            if path.exists():total+=sum(x['accounted_micro_yuan'] for x in read(path)['records'])/1e6
    return round(total,6)

def run(args):
    state=read(args.state);s=session(args.session);stop=args.state.parent/'STOP';lock=args.state.parent/'runner.lock'
    with lock.open('x',encoding='ascii') as stream:stream.write(str(os.getpid()))
    try:
        state.update(status='running',runner_pid=os.getpid(),started_at=state.get('started_at') or now());write(args.state,state)
        for index,unit in enumerate(state['units']):
            if unit.get('status')=='succeeded' and unit.get('qa',{}).get('source_scope_valid'):continue
            if stop.exists() or spent(state)>=state['cost_limit_yuan']:
                state.update(status='stopped',stop_reason='STOP flag' if stop.exists() else 'cost_limit',stopped_at=now());write(args.state,state);return
            if sha(unit['input_path'])!=unit['input_sha256']:raise RuntimeError('Queued input changed; refusing to upload')
            while True:
                existing=job_for(unit,state['course']['id'])
                if not existing:
                    with open(unit['input_path'],'rb') as document:
                        existing=post(s,'/api/upload',data={'course_id':state['course']['id']},files={'file':(unit['filename'],document,'application/pdf')})['job']
                    unit.update(job_id=existing['id'],attempts=1)
                else:
                    unit['job_id']=existing['id']
                    if existing['status'] in {'failed','cancelled'} and unit.get('attempts',1)<3:
                        existing=post(s,'/api/jobs/'+existing['id']+'/retry')['job'];unit['attempts']=unit.get('attempts',1)+1
                unit['status']='running';state['current_unit']=unit['id'];write(args.state,state);last=None
                while True:
                    job=read(ROOT/'runtime/jobs'/unit['job_id']/'job.json')
                    stage=(job['status'],job.get('stage'),job.get('message'))
                    if stage!=last:
                        unit.update(stage=job.get('stage'),progress=job.get('progress'),last_message=job.get('message'))
                        state['estimated_committed_yuan']=spent(state);write(args.state,state)
                        print(json.dumps({'unit':index+1,'total':len(state['units']),'file':unit['filename'],'status':job['status'],'stage':job.get('stage'),'message':job.get('message')},ensure_ascii=False),flush=True);last=stage
                    if stop.exists():
                        if job['status'] in {'queued','running','cancel_requested'}:post(s,'/api/jobs/'+unit['job_id']+'/cancel')
                        state.update(status='stopped',stop_reason='STOP flag',stopped_at=now());write(args.state,state);return
                    if job['status'] in {'succeeded','failed','cancelled'}:break
                    time.sleep(3)
                if job['status']=='succeeded':
                    unit['status']='succeeded';unit['database']=job.get('database');unit['graph_url']=job.get('graph_url')
                    break
                unit.update(status=job['status'],error=job.get('error'));write(args.state,state)
                if unit.get('attempts',1)>=3 or job['status']=='cancelled':break
                # Explicit batch recovery: same input/job, bounded at two retries.
            if unit['status']=='succeeded' and not unit.get('qa',{}).get('source_scope_valid') and not stop.exists():
                question='请只依据当前这段课件《'+unit['filename']+'》，概述主要知识点，并引用资料来源。'
                try:
                    response=s.post(BASE+'/api/qa/',json={'course_id':state['course']['id'],'job_id':unit['job_id'],'question':question},timeout=(10,180))
                    body=response.json();sources=body.get('sources',[])
                    scoped=bool(sources) and all(x.get('job_id')==unit['job_id'] for x in sources)
                    abstained=body.get('insufficient_evidence') is True and not sources
                    unit['qa']={'http_status':response.status_code,'question':question,'answer':body.get('answer'),'retrieval':body.get('retrieval'),
                                'insufficient_evidence':body.get('insufficient_evidence'),'sources':sources,
                                'source_scope_valid':response.status_code==200 and body.get('retrieval')=='course_pinecone_graph' and (scoped or abstained)}
                except requests.RequestException as error:
                    unit['qa']={'http_status':0,'error_type':type(error).__name__,'source_scope_valid':False}
                print(json.dumps({'qa_unit':index+1,'http_status':unit['qa']['http_status'],'source_scope_valid':unit['qa']['source_scope_valid']},ensure_ascii=False),flush=True)
            write(args.state,state)
        state.update(status='completed_with_failures' if any(u['status']!='succeeded' or not u.get('qa',{}).get('source_scope_valid') for u in state['units']) else 'completed',finished_at=now(),estimated_committed_yuan=spent(state))
        write(args.state,state)
    finally:
        lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','status','stop'])
    parser.add_argument('--folder',type=Path,default=Path(r'C:\Users\Lin\Desktop\study_documents\fdurop\ppt\ppt\ppt'))
    parser.add_argument('--state',type=Path,default=ROOT/'runtime/batch_20261101/queue.json')
    parser.add_argument('--session',type=Path,default=ROOT/'runtime/fullflow_test_20261101.json');args=parser.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='run':run(args)
    elif args.action=='stop':(args.state.parent/'STOP').write_text('Requested stop '+now(),encoding='utf-8')
    else:
        data=read(args.state);print(json.dumps({'status':data['status'],'course':data['course'],'sources':len(data['sources']),'units':len(data['units']),
            'completed':sum(u.get('status')=='succeeded' for u in data['units']),'current':next((u for u in data['units'] if u.get('id')==data.get('current_unit')),None),'estimated_committed_yuan':spent(data)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
