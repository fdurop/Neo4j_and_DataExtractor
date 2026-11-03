"""Bounded, cursor-paged Neo4j reads. A failed page never yields partial rows."""
import time
from neo4j.exceptions import ServiceUnavailable, SessionExpired


def read_pages(driver, query, course_id, batch_size=200):
    cursor = ""
    while True:
        for attempt in range(3):
            try:
                with driver.session(default_access_mode="READ") as session:
                    rows = list(session.run(query, {"course_id": course_id, "after": cursor, "limit": batch_size}))
                break
            except (ServiceUnavailable, SessionExpired):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        if not rows:
            return
        next_cursor = rows[-1]["cursor"]
        if not isinstance(next_cursor, str) or next_cursor <= cursor:
            raise RuntimeError("Graph read cursor failed to advance")
        print(f"Neo4j readback: {len(rows)} records received", flush=True)
        yield from rows
        cursor = next_cursor
        if len(rows) < batch_size:
            return
