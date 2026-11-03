"""Restore plain JSON envelopes before rolling back to a pre-native-CRUD server.

Read-only by default. Changes serialization only; never changes native entities.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'engines/graph'))
from app_config import load_settings
from neo4j_runtime import driver_for
from backend.services.native_graph import decode_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-id',required=True)
    parser.add_argument('--apply',action='store_true',help='Write plain JSON envelopes before checking out old code')
    args=parser.parse_args();cfg=load_settings()
    driver=driver_for(cfg['NEO4J_URI'],cfg['NEO4J_USER'],cfg['NEO4J_PASSWORD'],cfg['NEO4J_DATABASE'])
    try:
        with driver.session() as session:
            with session.begin_transaction() as tx:
                row=tx.run('MATCH (c:Course {id:$id}) RETURN c.review_graph_json AS review_graph_json,'
                           'c.model_graph_json AS model_graph_json,c.previous_review_graph_json AS previous_review_graph_json,'
                           'c.review_revision AS revision',id=args.course_id).single()
                if row is None:raise RuntimeError('Course not found')
                changes={}
                for key in ('review_graph_json','model_graph_json','previous_review_graph_json'):
                    value=row[key]
                    if value and json.loads(value).get('_graph_encoding')=='zlib-base64-v1':
                        changes[key]=json.dumps(decode_snapshot(value),ensure_ascii=False,allow_nan=False)
                if args.apply and changes:
                    updated=tx.run('MATCH (c:Course {id:$id}) WHERE c.review_revision=$revision '
                                   'SET c += $props RETURN c.id AS id',id=args.course_id,revision=row['revision'],props=changes).single()
                    if updated is None:raise RuntimeError('Revision changed; reload and retry')
                tx.commit()
        print(json.dumps({'course_id':args.course_id,'converted_fields':list(changes),'applied':args.apply},ensure_ascii=False))
    finally:driver.close()


if __name__=='__main__':main()
