import hashlib,json,tempfile,unittest
from pathlib import Path
from backend.services.pipeline_checkpoints import budget_path,graph_checkpoint

class CheckpointTests(unittest.TestCase):
    def setup_phase(self,folder,status='success'):
        job=folder/'job';job.mkdir();kg=job/'extraction/kg_data';kg.mkdir(parents=True)
        (kg/'pages.json').write_text('[]');pdf=job/'source.pdf';pdf.write_bytes(b'pdf')
        phase=job/'graph';phase.mkdir();scope={'extraction_run_id':'job-id','source_pdf_sha256':hashlib.sha256(b'pdf').hexdigest(),'input_file_sha256':{'pages.json':hashlib.sha256(b'[]').hexdigest()}}
        (phase/'validation_scope.json').write_text(json.dumps(scope));(phase/'summary.json').write_text(json.dumps({'status':status,'input_files_unchanged':True}))
        (phase/'api_results.json').write_text('[{"response":"paid result"}]')
        return job,kg,pdf,phase

    def test_completed_phase_reused_without_new_directory(self):
        with tempfile.TemporaryDirectory() as d:
            job,kg,pdf,phase=self.setup_phase(Path(d));selected,summary=graph_checkpoint(job,kg,'job-id',pdf)
            self.assertEqual(selected,phase);self.assertEqual(summary['status'],'success');self.assertFalse(list(job.glob('graph_retry_*')))

    def test_changed_source_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as d:
            job,kg,pdf,phase=self.setup_phase(Path(d));pdf.write_bytes(b'other')
            with self.assertRaises(ValueError):graph_checkpoint(job,kg,'job-id',pdf)

    def test_changed_material_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as d:
            job,kg,pdf,phase=self.setup_phase(Path(d));(kg/'pages.json').write_text('[1]')
            with self.assertRaises(ValueError):graph_checkpoint(job,kg,'job-id',pdf)

    def test_failed_phase_preserved_and_paid_response_cache_copied(self):
        with tempfile.TemporaryDirectory() as d:
            job,kg,pdf,phase=self.setup_phase(Path(d),'incomplete');before=(phase/'api_results.json').read_bytes()
            selected,summary=graph_checkpoint(job,kg,'job-id',pdf)
            self.assertIsNone(summary);self.assertEqual(selected.name,'graph_retry_001');self.assertEqual((selected/'api_results.json').read_bytes(),before);self.assertEqual((phase/'api_results.json').read_bytes(),before)

    def test_prior_global_ledger_kept_when_enabling_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=Path(d);job=runtime/'jobs/job';log=job/'extraction/api_calls.json';log.parent.mkdir(parents=True)
            expected=runtime/'api_budget.json';log.write_text(json.dumps([{'budget_ledger_path':str(expected)}]))
            self.assertEqual(budget_path(job,runtime,True),expected.resolve())

    def test_other_jobs_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=Path(d);job=runtime/'jobs/job';log=job/'extraction/api_calls.json';log.parent.mkdir(parents=True)
            log.write_text(json.dumps([{'budget_ledger_path':str(runtime/'jobs/other/api_budget.json')}]))
            with self.assertRaises(ValueError):budget_path(job,runtime,True)

if __name__=='__main__':unittest.main()
