# Interactive graph editor

Branch: graph/interactive-editor-20261031
Rollback tag: graph-before-20261031

- Force-directed draggable circular nodes; click nodes for data and source evidence.
- Click an edge to edit its type/description or delete it.
- Add nodes and relationships from the toolbar, or choose a node and start a connection.
- Right-click a node to pin/unpin; pause neighbor movement; relayout and zoom controls.
- Layout positions are remembered in this browser for this course. They are not shared across users.
- Saved mutations use the existing reviewed snapshot in Neo4j. Original extracted Concept nodes and native relationships are preserved; this is not a migration of the original graph.
- Undo restores one previous saved snapshot. Revisions reject stale submissions from other tabs.
- Existing PDF/source links require the source files and job records to be present locally.

Validation: 134 Python regression tests; headless Cytoscape layout/selection/pinning check;
isolated Edge tests with a read-only copy of the 205-node cloud graph for rendering,
dragging, editing, undo and edge selection. No test mutations were sent to the cloud course.

To roll back source: git switch local-baseline
Restart start.cmd after switching branches, then refresh the browser.

Additional isolated browser check: creating/deleting both nodes and relationships passed using a temporary fixture. No horizontal page overflow at 390px.


## Evidence and spring update
Source evidence and raw text are collapsed by default. Dragging uses a bounded, damped spring simulation over at most 60 nearby nodes, settling after release; fixed nodes and reduced-motion preferences are respected. The standalone read-only verification script in scripts/verify_graph_delete.py checks both reviewed snapshots and native Neo4j nodes. See scripts/VERIFY_GRAPH_DELETE.md.


## One-hop focus and compact inspector
Click a node to highlight its incident edges and direct neighbors; other nodes and edges fade, including edges between neighbors. Click the same node again or click blank canvas to clear the selection. Focus does not relayout nodes. The inspector warns before discarding unsaved text edits.

The description field starts at one visual line, grows with wrapping/input to 240px, then scrolls. Relation controls follow the compact name/description/save area. More than eight relationships are grouped by relation type. IDs, teaching-role controls, learning-path details and source evidence are secondary/collapsible. Relation labels are off by default and appear for focused edges.

Rollback for this update: graph-before-focus-panel-20261031.
Validation: 139 Python regressions; Cytoscape one-hop/toggle/background/drag/position tests; spring tests; isolated Edge checks for focus, neighbor-edge dimming, textarea growth/shrink, relation ordering/grouping, unsaved-edit confirmation, CRUD and undo. Cloud course data was not modified by these checks.
