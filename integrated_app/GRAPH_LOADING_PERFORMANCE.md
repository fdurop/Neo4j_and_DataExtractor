# Graph loading performance, 2026-11-02

The graph page now requests `/api/graph?view=map`. The original full endpoint
remains available for export, retrieval and editing internals. Both use the
native Neo4j graph; a lightweight map is not a separate editable graph.

## Changes

- Query only drawing fields rather than every node/relationship property.
- Fetch a map in large driver batches instead of dozens of full-property reads.
- Reuse the repository's Neo4j driver and connection pool; close it on shutdown.
- Avoid downloading the entire 1.68 MB compressed review envelope each time.
  Only graph-level audit/organization metadata uses a cache, checked against the
  live course revision. It is refreshed on a revision mismatch, and never supplies
  native topology or editable node values.
- Read actual nodes and relationships on every map load. Check the revision again
  afterward; retry once if a web edit committed during the read.
- Map reads do not acquire the long-running mutation/vector synchronization lock.
- Compress map/detail HTTP responses and use no-store, so a reload reaches the DB.
- Load evidence, source text and detailed node metadata when its details section
  is expanded. Ignore stale responses and preserve unsaved form drafts.
- Fetch the complete original endpoint when exporting JSON.

## Measurements on the local computer

Course: `course_3ad732e2e6744922b29ea4c85e6566d6`.
The user's current course at measurement time had 2238 nodes / 7639 relationships.

| Measurement | Before | After |
| --- | --- | --- |
| Data request | 37.528 s | 2.331–10.315 s across observed runs |
| Data sent to browser | 17,597,985 bytes | 390,434 bytes with gzip |
| Actual page ready | Not measured end-to-end before | 6.786 s and 9.401 s in two live runs |
| Evidence expanded | Included in every full load | About 4.2 s on demand |

Before: rendering with the captured real response took 4.128 s; this was a
separate renderer measurement, not a live end-to-end page test. The final-version
run measured 10.315 s for a cold request, 7.685 s for the next request and 9.401 s
for the actual page. Network timing varies; these are observations, not guarantees.

Compared both real responses: node IDs and relationship tuples match exactly;
normalized concept descriptions, importance, teaching roles, review states and
source page/document positions also match. The initial projection omits heavy
source/model metadata, not graph entities. Complete export remains separate.

The actual UI also verified source expansion and draft preservation. A disposable
node was created through the page, read from native Neo4j, displayed in the editor
after the fast reload and deleted through the same page. This does not optimize
the full native mutation/history/vector pipeline itself; saves and full JSON
exports can still take longer than opening the map.

Validation: 181 Python checks and the existing focus/spring JavaScript checks
passed. Local evidence is under the operator's `outputs/graph-speed-20261102`.

Rollback of this performance change: parent commit `1638034`; no new graph schema
or course-data migration is required to return to that native-CRUD version.
This work updates the local computer only; the Windows server needs the same code
update to gain the speed improvement.
