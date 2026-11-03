"""Small live graph projection for drawing; full native data stays available on demand."""
import copy
from backend.routes.graph_contract import GraphError
from .native_graph import LABELS, SNAPSHOTS

NODE_FIELDS = ('id','name','canonical_name','node_type','layer','category','importance_level',
    'aliases','review_status','origin','suggested_teaching_role','teaching_role','role_review_status',
    'evidence_node_ids','source_page_ids','chapter_id','document_id','page_id','page_no','title',
    'file_name','source_page_id','job_id','previous_names','model_update_pending')
LINK_FIELDS = ('id','relation_id','relationship_id','description','review_status','auto_generated',
               'origin','review_required')


def _projection(variable, fields):
    return variable+' {'+', '.join('.'+x for x in fields)+'}'


def node_record(labels, properties, full=False):
    values={k:v for k,v in dict(properties).items() if k not in SNAPSHOTS and v is not None}
    kind=values.pop('_app_type',None) or next((k for k,v in LABELS.items() if v in labels),'material')
    record={'id':values['id'],'type':kind,'labels':list(labels),'properties':values,
            'name':values.get('name') or values.get('canonical_name') or values.get('title') or values.get('id'),
            'description':values.get('description','')}
    if full and 'Concept' in labels:
        import json
        for field in ('page_observations','previous_names'):
            if field+'_json' in values:
                try:values[field]=json.loads(values[field+'_json'])
                except (TypeError,ValueError):pass
    return record


def link_record(source,target,kind,properties):
    return {'source':source,'target':target,'relation':kind,'properties':dict(properties)}


def project_stored(graph):
    """Legacy local/snapshot mode keeps its existing semantics with a smaller wire format."""
    result={k:copy.deepcopy(v) for k,v in graph.items() if k not in {'nodes','links'}}
    result['nodes']=[]
    for node in graph.get('nodes',[]):
        values={**(node.get('properties') or {}),**{k:v for k,v in node.items() if k!='properties'}}
        fields=(*NODE_FIELDS,'description') if node.get('type') in {'concept','knowledge'} or 'Concept' in node.get('labels',[]) else NODE_FIELDS
        result['nodes'].append({'id':node['id'],'name':node.get('name',node['id']),
            'type':node.get('type'),'labels':node.get('labels',[]),
            'properties':{k:values[k] for k in fields if k in values}})
    result['links']=[{**{k:l[k] for k in ('source','target','relation','value') if k in l},
        'properties':{k:v for k,v in {**(l.get('properties') or {}),**l}.items() if k in LINK_FIELDS}}
        for l in graph.get('links',[])]
    result['projection']='map-v1'
    return result


def read_map(session,course_id,progress=None):
    """Three bounded-size projections, not dozens of full-property page round trips."""
    total_nodes=total_links=0
    if progress:
        progress(phase="counting")
        count=session.run('MATCH (n {course_id:$course}) WHERE n.id IS NOT NULL WITH count(n) AS nodes '
            'OPTIONAL MATCH (a {course_id:$course})-[r]->(b {course_id:$course}) '
            'WHERE a.id IS NOT NULL AND b.id IS NOT NULL RETURN nodes,count(r) AS links',course=course_id).single()
        total_nodes,total_links=count['nodes'],count['links']
        progress(phase="nodes",nodes=0,links=0,total_nodes=total_nodes,total_links=total_links)
    rows=session.run('MATCH (n {course_id:$course}) WHERE n.id IS NOT NULL RETURN labels(n) AS labels, '
        +_projection('n',NODE_FIELDS)+ ' + {description:CASE WHEN n:Concept THEN n.description ELSE null END,'
        ' _app_type:n._app_type} AS props',course=course_id)
    nodes=[]
    for row in rows:
        nodes.append(node_record(row['labels'],row['props']))
        if progress and len(nodes)%100==0:
            progress(phase="nodes",nodes=len(nodes))
    if progress:progress(phase="links",nodes=len(nodes))
    rows=session.run('MATCH (a {course_id:$course})-[r]->(b {course_id:$course}) '
        'WHERE a.id IS NOT NULL AND b.id IS NOT NULL RETURN a.id AS source,b.id AS target,'
        'type(r) AS kind,'+_projection('r',LINK_FIELDS)+' AS props',course=course_id)
    links=[];seen=set()
    for row in rows:
        key=row['source'],row['target'],row['kind']
        if key in seen:raise GraphError('数据库存在重复关系，请先处理',409)
        seen.add(key);links.append(link_record(*key,row['props']))
        if progress and len(links)%200==0:progress(phase="links",links=len(links))
    if progress:progress(phase="checking",links=len(links),nodes=len(nodes))
    return nodes,links


def read_details(session,course_id,node_id):
    row=session.run('MATCH (n {course_id:$course,id:$id}) RETURN labels(n) AS labels,'
        'n {.*,review_graph_json:null,model_graph_json:null,previous_review_graph_json:null} AS props',
        course=course_id,id=node_id).single()
    if row is None:raise GraphError('节点已不存在，请重新加载图谱',404)
    node=node_record(row['labels'],row['props'],True)
    props=node['properties']
    evidence=set(props.get('evidence_node_ids') or [])
    rows=list(session.run('MATCH (a {course_id:$course,id:$id})-[r]-(b {course_id:$course}) '
        'RETURN startNode(r).id AS source,endNode(r).id AS target,type(r) AS kind,properties(r) AS props,'
        'b.id AS peer',course=course_id,id=node_id))
    evidence.update(r['peer'] for r in rows if r['kind']=='HAS_EVIDENCE' and r['source']==node_id)
    pages=set(props.get('source_page_ids') or [])
    records=[node]
    if evidence or pages:
        rows2=session.run('MATCH (n {course_id:$course}) WHERE n.id IN $ids OR n.id IN $pages '
            'RETURN labels(n) AS labels,properties(n) AS props',course=course_id,ids=list(evidence),pages=list(pages))
        records.extend(node_record(r['labels'],r['props'],True) for r in rows2)
    loaded={n['id'] for n in records}
    extra={n['properties'].get('source_page_id') or n['properties'].get('page_id') for n in records}
    extra.discard(None);extra-=loaded
    if extra:
        records.extend(node_record(r['labels'],r['props'],True) for r in session.run(
            'MATCH (n:Page {course_id:$course}) WHERE n.id IN $ids RETURN labels(n) AS labels,properties(n) AS props',
            course=course_id,ids=list(extra)))
    return {'nodes':records,'links':[link_record(r['source'],r['target'],r['kind'],r['props']) for r in rows],
            'details_for':node_id}


def stored_details(graph,node_id):
    by_id={n['id']:n for n in graph.get('nodes',[])}
    node=by_id.get(node_id)
    if not node:raise GraphError('节点已不存在，请重新加载图谱',404)
    values={**(node.get('properties') or {}),**node}
    ids={node_id,*values.get('evidence_node_ids',[]),*values.get('source_page_ids',[])}
    links=[r for r in graph.get('links',[]) if r.get('source')==node_id or r.get('target')==node_id]
    ids.update(r['target'] for r in links if r.get('source')==node_id and r.get('relation')=='HAS_EVIDENCE')
    for identity in list(ids):
        record=by_id.get(identity,{})
        p={**(record.get('properties') or {}),**record}
        if p.get('source_page_id'):ids.add(p['source_page_id'])
    return {'nodes':[by_id[x] for x in ids if x in by_id],'links':links,'details_for':node_id}
