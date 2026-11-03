"""Same-job checkpoints for bounded batch processing; never relabel old inputs."""
import hashlib
import json
from pathlib import Path
import shutil

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def budget_path(directory, runtime, isolated=False):
    directory, runtime = Path(directory).resolve(), Path(runtime).resolve()
    permitted = {runtime / 'api_budget.json', directory / 'api_budget.json'}
    logs = [directory / 'extraction/api_calls.json', *directory.glob('graph*/api_results.json')]
    for path in logs:
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding='utf-8'))
        if records:
            original = Path(records[0]['budget_ledger_path']).resolve()
            if original not in permitted:
                raise ValueError('调用记录的预算账本不属于当前任务')
            return original
    return directory / 'api_budget.json' if isolated else runtime / 'api_budget.json'

def graph_checkpoint(directory, kg_data, job_id, pdf):
    directory, kg_data = Path(directory), Path(kg_data)
    phases = [p for p in directory.iterdir() if p.is_dir() and
              (p.name == 'graph' or (p.name.startswith('graph_retry_') and p.name[12:].isdigit()))]
    if not phases:
        return directory / 'graph', None
    phases.sort(key=lambda p: 0 if p.name == 'graph' else int(p.name[12:]))
    previous = phases[-1]
    scope_file = previous / 'validation_scope.json'
    if not scope_file.exists():
        if not any(p.name != 'api_results.json' for p in previous.iterdir()):
            return previous, None
        raise ValueError('已有建图目录缺少来源记录，不覆盖')
    scope = json.loads(scope_file.read_text(encoding='utf-8'))
    if scope.get('extraction_run_id') != job_id or scope.get('source_pdf_sha256') != digest(pdf):
        raise ValueError('恢复任务与原PDF或任务身份不一致')
    if any(digest(kg_data / name) != value for name, value in scope['input_file_sha256'].items()):
        raise ValueError('恢复任务的预处理输入发生变化')
    summary_file = previous / 'summary.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8')) if summary_file.exists() else None
    if summary and summary.get('status') == 'success' and summary.get('input_files_unchanged') is True:
        return previous, summary
    number = 1 if previous.name == 'graph' else int(previous.name[12:]) + 1
    target = directory / f'graph_retry_{number:03d}'
    target.mkdir()
    cache = previous / 'api_results.json'
    if cache.exists():
        shutil.copy2(cache, target / cache.name)
    return target, None

def stamp_imported_job(driver, course_id, job_id):
    with driver.session() as session:
        session.run('MATCH (n {course_id:$course_id}) '
                    'WHERE n.data_version=$job_id OR n.latest_model_data_version=$job_id '
                    'SET n.job_id=$job_id', course_id=course_id, job_id=job_id).consume()
