"""Course-scoped Pinecone indexing with one shared embedding model.

Reviewed text is checked before each question; edits/deletions are synchronized
before retrieval. Provider errors fail explicitly; no silent keyword fallback.
"""
import hashlib
import json
from pathlib import Path
from threading import RLock, local
import time
import requests

_lock = RLock()
_transport = local()

def _session():
    if not hasattr(_transport, "session"):
        _transport.session = requests.Session()
    return _transport.session
KINDS = {'page', 'text_chunk', 'figure', 'table', 'formula', 'code_block', 'concept', 'knowledge'}

def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def _post(url, headers, payload):
    # Embedding/query are read operations; upsert/delete use stable IDs/filters.
    # A lost response can therefore be retried without duplicate vector rows.
    for attempt in range(3):
        try:
            response = _session().post(url, headers=headers, json=payload, timeout=(10, 90), allow_redirects=False)
        except (requests.Timeout, requests.ConnectionError) as error:
            permanent = 'certificate_verify_failed' in str(error).lower()
            if permanent or attempt == 2:
                raise RuntimeError('向量服务网络连接失败（'+type(error).__name__+'），请稍后重试处理') from None
        except requests.RequestException as error:
            raise RuntimeError('向量服务请求失败（'+type(error).__name__+'）') from None
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in {429, 502, 503, 504} or attempt == 2:
                raise RuntimeError(f'向量服务请求失败 HTTP {response.status_code}')
        time.sleep(attempt + 1)


def _write_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)

def embed(settings, texts, text_type):
    model = settings['EMBEDDING_MODEL']
    dimension = int(settings['EMBEDDING_DIMENSION'])
    body = _post('https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding',
        {'Authorization': 'Bearer ' + settings['DASHSCOPE_API_KEY']},
        {'model': model, 'input': {'texts': texts},
         'parameters': {'dimension': dimension, 'text_type': text_type}})
    rows = sorted(body.get('output', {}).get('embeddings', []), key=lambda x: x['text_index'])
    if len(rows) != len(texts) or any(len(x.get('embedding', [])) != dimension for x in rows):
        raise RuntimeError('Embedding数量或维度与配置不一致')
    return [x['embedding'] for x in rows]

def records_for(graph, course_id):
    records = []
    integration = graph.get('integration', {})
    for node in graph.get('nodes', []):
        props = node.get('properties') or {}
        kind = str(node.get('type') or props.get('node_type') or '').lower()
        if kind not in KINDS:
            continue
        identity = str(node.get('id') or '')
        if not identity:
            continue
        if props.get('course_id') and props['course_id'] != course_id:
            raise ValueError('拒绝索引其他课程节点')
        corrected = props.get('origin') == 'teacher_edit'
        chunks = [node.get('name', '')]
        chunks += [node.get('description', '')] if corrected else [
            props.get('source_text', ''), props.get('raw_text', ''),
            props.get('embedding_text', ''), node.get('description', '')]
        text = '\n'.join(dict.fromkeys(str(v).strip() for v in chunks if str(v or '').strip()))[:6000]
        if not text:
            continue
        metadata = {'course_id': course_id, 'node_id': identity, 'node_type': kind,
                    'text_hash': _hash(text), 'text': text[:1800]}
        for key in ['document_id', 'page_id', 'source_page_id', 'job_id']:
            value = props.get(key) or node.get(key) or (integration.get('job_id') if key == 'job_id' else None)
            if value:
                metadata[key] = str(value)
        if props.get('page_no') is not None:
            metadata['page_no'] = int(props['page_no'])
        records.append({'id': 'course_' + _hash(course_id + ':' + identity), 'text': text, 'metadata': metadata})
    return records

def _connection(settings):
    key = settings.get('PINECONE_API_KEY')
    name = settings.get('PINECONE_INDEX_NAME')
    if not key or not name:
        raise RuntimeError('Pinecone配置缺失')
    headers = {'Api-Key': key, 'X-Pinecone-Api-Version': '2025-04'}
    try:
        response = _session().get('https://api.pinecone.io/indexes/' + name, headers=headers,
                                timeout=(10, 30), allow_redirects=False)
    except requests.RequestException as error:
        raise RuntimeError('Pinecone索引连接失败（'+type(error).__name__+'）') from None
    if response.status_code != 200:
        raise RuntimeError(f'Pinecone索引读取失败 HTTP {response.status_code}')
    info = response.json()
    host = info.get('host', '')
    if not host.endswith('.pinecone.io') or any(c in host for c in '/:@?#'):
        raise RuntimeError('Pinecone索引主机无效')
    if info.get('dimension') != int(settings['EMBEDDING_DIMENSION']) or info.get('metric') != 'cosine':
        raise RuntimeError('Pinecone索引维度或距离度量不匹配')
    return 'https://' + host, headers

def sync_graph(settings, graph, course_id):
    with _lock:
        records = records_for(graph, course_id)
        signature = {'schema_version': 2, 'model': settings['EMBEDDING_MODEL'], 'dimension': int(settings['EMBEDDING_DIMENSION']),
                     'index': settings['PINECONE_INDEX_NAME'], 'namespace': settings.get('PINECONE_NAMESPACE', ''),
                     'records': {x['id']: _hash(json.dumps(x['metadata'], ensure_ascii=False, sort_keys=True)) for x in records}}
        folder = Path(settings['APP_DATA_DIR']) / 'vector_indexes' / _hash(course_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'manifest.json'
        previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if previous == signature:
            return {'index': signature['index'], 'record_count': len(records), 'changed_count': 0, 'deleted_count': 0}
        host, headers = _connection(settings)
        compatible = all(previous.get(k) == signature[k] for k in ['model', 'dimension', 'index', 'namespace'])
        previous_records = previous.get('records', {}) if compatible else {}
        pending_path = folder / 'pending_sync.json'
        pending = json.loads(pending_path.read_text(encoding='utf-8')) if pending_path.exists() else {}
        if all(pending.get(k) == signature[k] for k in ['model', 'dimension', 'index', 'namespace']):
            previous_records = {**previous_records, **pending.get('records', {})}
        changes = [x for x in records if previous_records.get(x['id']) != signature['records'][x['id']]]
        deleted = list(set(previous_records) - set(signature['records']))
        batches = []
        checkpoint = {k: signature[k] for k in ['model', 'dimension', 'index', 'namespace']}
        checkpoint['records'] = dict(previous_records)
        embedding_path = folder / 'embedding_batch.json'
        for start in range(0, len(changes), 8):
            batch = changes[start:start + 8]
            cache_key = _hash(json.dumps([signature['model'], signature['dimension'], [x['text'] for x in batch]], ensure_ascii=False))
            cached = json.loads(embedding_path.read_text(encoding='utf-8')) if embedding_path.exists() else {}
            if cached.get('key') == cache_key:
                vectors = cached['vectors']
            else:
                vectors = embed(settings, [x['text'] for x in batch], 'document')
                _write_checkpoint(embedding_path, {'key': cache_key, 'vectors': vectors})
            rows = [{'id': item['id'], 'values': vector,
                     'metadata': {**item['metadata'], 'embedding_model': signature['model']}}
                    for item, vector in zip(batch, vectors)]
            _post(host + '/vectors/upsert', headers, {'namespace': signature['namespace'], 'vectors': rows})
            batches.extend(rows)
            checkpoint['records'].update({x['id']: signature['records'][x['id']] for x in batch})
            _write_checkpoint(pending_path, checkpoint)
        if deleted:
            _post(host + '/vectors/delete', headers, {'namespace': signature['namespace'], 'ids': deleted})
        # Check actual stored metadata after eventual-consistency propagation.
        if records:
            sample = changes[-1] if changes else records[-1]
            for attempt in range(8):
                try:
                    response = _session().get(host + '/vectors/fetch', headers=headers,
                        params={'namespace': signature['namespace'], 'ids': sample['id']},
                        timeout=(10, 30), allow_redirects=False)
                    stored = response.json().get('vectors', {}).get(sample['id'], {}) if response.status_code == 200 else {}
                except requests.RequestException:
                    stored = {}
                if all(stored.get('metadata', {}).get(k) == v for k, v in {**sample['metadata'], 'embedding_model': signature['model']}.items()):
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Pinecone写入后读回未确认，尚未标记同步成功')
        (folder / 'last_upsert.json').write_text(json.dumps(batches, ensure_ascii=False), encoding='utf-8')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
        pending_path.unlink(missing_ok=True)
        embedding_path.unlink(missing_ok=True)
        return {'index': signature['index'], 'record_count': len(records),
                'changed_count': len(changes), 'deleted_count': len(deleted),
                'model': signature['model'], 'dimension': signature['dimension']}

def search(settings, graph, course_id, question, limit=12, job_id=None):
    sync_graph(settings, graph, course_id)
    host, headers = _connection(settings)
    vector = embed(settings, [question], 'query')[0]
    conditions = [{'course_id': {'$eq': course_id}}, {'embedding_model': {'$eq': settings['EMBEDDING_MODEL']}}]
    if job_id:
        conditions.append({'job_id': {'$eq': job_id}})
    result = _post(host + '/query', headers, {'namespace': settings.get('PINECONE_NAMESPACE', ''),
        'vector': vector, 'topK': limit, 'includeMetadata': True, 'includeValues': False,
        'filter': {'$and': conditions}})
    return [x.get('metadata', {}).get('node_id') for x in result.get('matches', [])]
