# Native graph editing

Neo4j courses are migrated on their first successful save. The existing reviewed
graph is reconciled against the original model; previous review deletions are
applied to actual nodes and relationships. Other courses are not migrated by reads.

For migrated courses, node/relationship existence and editable properties are
read from native Neo4j entities, not from the JSON review envelope. All writes are
scoped by course_id and stable id. Changes, review revision, compressed history and
the review envelope commit in one transaction. A version conflict rolls back.
Undo restores both native entities and the review envelope. Read-only network
retries are bounded; writes are not automatically retried.

The original model and compressed GraphRevision history remain available.
History uses owner_course_id rather than course_id so audit records are not shown
as course knowledge nodes. Snapshot envelopes support both legacy plain JSON and
zlib/base64 JSON wrappers; scripts/verify_graph_delete.py reads both.

After a successful edit, Pinecone is synchronized. API vector_sync.status is
synced, disabled or pending. Pending does not roll back an already committed Neo4j
transaction: the UI warns, and question retrieval retries synchronization before
using vectors. External Neo4j changes become visible on reload and are synchronized
before the next question. Neo4j and Pinecone do not share a distributed transaction.

Publishing uses the last durable review for three-way merging, so the importer's
earlier native writes cannot erase teacher changes or resurrect saved deletions.
Parallel relationships with the same direction/type are explicitly rejected rather
than deleting an arbitrary match; different types remain separate.

## Rollback and deployment

Commit d837a38 is the code checkpoint before this implementation. Git rollback
does not undo database edits. The server needs the updated code to read compressed
envelopes. Before reverting to older code, run this utility with the new code still
checked out (repeat for each migrated course):

```powershell
python scripts/restore_snapshot_json.py --course-id COURSE_ID
python scripts/restore_snapshot_json.py --course-id COURSE_ID --apply
```

The first command only inspects; --apply restores plain JSON serialization without
altering graph content or revisions. After that, old code can read the snapshots,
but its editing behavior remains snapshot-only. Do not run old and new editing
servers concurrently against the same course.

## Local verification, 2026-11-02

Course: course_3ad732e2e6744922b29ea4c85e6566d6.
Real Edge forms exercised editing an existing node, undo, creating a temporary
node, creating a relationship, changing its native type, deleting the relationship
and deleting the temporary node. Each step was read back with native Cypher.
Neo4j Aura's open Query page independently showed the modified description,
created node, RELATED_TO then REQUIRES, and zero counts after deletion.
Pinecone was independently fetched before and after temporary-node deletion.
Final state: 2239 native nodes, 7643 native relationships, matching web counts;
original description restored; temporary node, relationship and vector absent.
Seven durable revisions were recorded. A transient read failure and two pending
vector synchronizations were observed and recovered; the large course still takes
tens of seconds to load/save on the current connection. No extraction was rerun.
