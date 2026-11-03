import unittest
from unittest.mock import MagicMock, Mock, patch
from engines.graph.neo4j_runtime import run_idempotent_write
from neo4j.exceptions import SessionExpired, AuthError
from backend.services.pipeline_recovery import retry_model_transport


class ImportRecoveryTests(unittest.TestCase):
    def test_failed_batch_reopens_session_and_consumes_before_commit(self):
        driver=MagicMock();session=driver.session.return_value.__enter__.return_value
        tx=MagicMock();tx.run.return_value.consume.side_effect=[SessionExpired('lost'), 'done']
        session.execute_write.side_effect=lambda work:work(tx)
        with patch('engines.graph.neo4j_runtime.time.sleep'):
            self.assertEqual(run_idempotent_write(driver,'MERGE (n {id:$id})',{'id':'a'}),'done')
        self.assertEqual(driver.session.call_count,2)
        self.assertEqual(tx.run.call_args.args,('MERGE (n {id:$id})',{'id':'a'}))

    def test_permanent_database_failure_is_not_retried(self):
        driver=MagicMock();driver.session.side_effect=AuthError('invalid')
        with self.assertRaises(AuthError):run_idempotent_write(driver,'MERGE (n)')
        self.assertEqual(driver.session.call_count,1)

    def test_database_retries_are_bounded(self):
        driver=MagicMock();driver.session.side_effect=SessionExpired('offline')
        with patch('engines.graph.neo4j_runtime.time.sleep'),self.assertRaises(SessionExpired):
            run_idempotent_write(driver,'MERGE (n)')
        self.assertEqual(driver.session.call_count,3)

    def test_tls_retry_uses_budgeted_operation_and_is_bounded(self):
        operation=Mock(side_effect=[RuntimeError('SSLEOFError unexpected_eof_while_reading'),{'ok':True}])
        with patch('backend.services.pipeline_recovery.time.sleep'):
            self.assertEqual(retry_model_transport(operation),{'ok':True})
        self.assertEqual(operation.call_count,2)
        operation=Mock(side_effect=RuntimeError('HTTP 503'))
        with patch('backend.services.pipeline_recovery.time.sleep'),self.assertRaises(RuntimeError):retry_model_transport(operation)
        self.assertEqual(operation.call_count,2)

    def test_budget_json_and_auth_errors_do_not_retry(self):
        for error in (RuntimeError('HTTP 401'),RuntimeError('Budget exhausted'),ValueError('bad JSON')):
            operation=Mock(side_effect=error)
            with self.assertRaises(type(error)):retry_model_transport(operation)
            self.assertEqual(operation.call_count,1)


if __name__=='__main__':unittest.main()
