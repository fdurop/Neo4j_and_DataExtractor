from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from neo4j.exceptions import SessionExpired

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"engines/graph"))
from graph_readback import read_pages


class ReadbackTests(unittest.TestCase):
    def test_partial_failed_page_is_retried_without_loss_or_duplicates(self):
        class Driver:
            calls=[]
            failed=False
            def session(self, **kwargs):
                self.mode=kwargs["default_access_mode"]
                return self
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def run(self,query,params):
                self.calls.append(dict(params))
                rows=[{"cursor":f"{i:03}","value":i} for i in range(6) if f"{i:03}">params["after"]][:params["limit"]]
                if params["after"]=="001" and not self.failed:
                    self.failed=True
                    def interrupted():
                        yield rows[0]
                        raise SessionExpired("interrupted")
                    return interrupted()
                return rows
        driver=Driver()
        with patch("graph_readback.time.sleep"):
            rows=list(read_pages(driver,"READ QUERY","course-a",batch_size=2))
        self.assertEqual([r["value"] for r in rows],list(range(6)))
        self.assertEqual([p["after"] for p in driver.calls],["","001","001","003","005"])
        self.assertTrue(all(p["course_id"]=="course-a" for p in driver.calls))
        self.assertEqual(driver.mode,"READ")

    def test_network_retries_are_bounded(self):
        class Driver:
            calls=0
            def session(self,**kwargs): return self
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def run(self,*args):
                self.calls+=1
                raise SessionExpired("offline")
        d=Driver()
        with patch("graph_readback.time.sleep"),self.assertRaises(SessionExpired):
            list(read_pages(d,"READ QUERY","c"))
        self.assertEqual(d.calls,3)

    def test_non_network_errors_are_not_retried(self):
        class Driver:
            calls=0
            def session(self,**kwargs): return self
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def run(self,*args):
                self.calls+=1
                raise ValueError("invalid query")
        d=Driver()
        with self.assertRaises(ValueError): list(read_pages(d,"INVALID","c"))
        self.assertEqual(d.calls,1)

if __name__=="__main__": unittest.main()
