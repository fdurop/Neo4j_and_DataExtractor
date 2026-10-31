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
