"""Exercise the shipped graph renderer with Cytoscape's headless viewport.

Only the DOM tooltip, viewport dimensions and animation-frame queue are supplied
by the harness. Node insertion, selection, zoom/pan events, styles and the public
renderer API use the actual local JavaScript implementation.
"""

from pathlib import Path
import json
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "frontend" / "static" / "js"
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node.js is required for renderer interaction checks")
class GraphCanvasInteractionTests(unittest.TestCase):
    def run_renderer(self, assertions):
        modules = "\n".join(
            "const %s = require(%s);" % (name, json.dumps(str(JS / filename)))
            for name, filename in (
                ("M", "knowledge_graph_model.js"),
                ("L", "graph_map_layout.js"),
                ("C", "graph_canvas.js"),
                ("cytoscape", "vendor/cytoscape.min.js"),
            )
        )
        harness = r"""
const assert = require('node:assert/strict');
let cy, frames = [], lastInfo = null, relativeZoom = null;
const notifications = [];
global.document = {createElement() { return {hidden: true, style: {}}; }};
global.requestAnimationFrame = callback => { frames.push(callback); return frames.length; };
function flush() {
  let remaining = 100;
  while (frames.length && remaining-- > 0) frames.shift()();
  assert.ok(remaining > 0, 'Viewport updates must settle without an animation loop');
}
const renderer = C.create({append() {}, clientWidth: 1265, clientHeight: 606}, {
  cytoscape(options) {
    cy = cytoscape({...options, container: undefined, headless: true, styleEnabled: true});
    cy.width = () => 1265;
    cy.height = () => 606;
    return cy;
  },
  model: M,
  mapLayout: L,
  onVisible(info) { lastInfo = info; notifications.push(info); },
  onZoom(value) { relativeZoom = value; }
});
function concept(id, tier, documentId = 'document-a') {
  return {id, name: id, type: 'knowledge',
    importance_level: tier === 1 || tier === 3 ? 1 : 2,
    suggested_teaching_role: tier >= 3 ? 'detail' : 'learning',
    sources: [{document_id: documentId, page_id: documentId + ':' + id, page_no: tier}]};
}
function fixture() {
  return M.normalize({documents: [{document_id: 'document-a'}, {document_id: 'document-b'}],
    nodes: [concept('tier1', 1), concept('tier2', 2), concept('tier3', 3),
      concept('tier4', 4, 'document-b')],
    links: [
      {id: 'e12', source: 'tier1', target: 'tier2', relation: 'DRIVES'},
      {id: 'e23', source: 'tier2', target: 'tier3', relation: 'REQUIRES'},
      {id: 'e34', source: 'tier3', target: 'tier4', relation: 'PART_OF'}]});
}
function viewOf(graph, filters = {}) {
  return M.selectView(graph, {mode: 'overview', showDetails: true, ...filters});
}
function render(view, selected) {
  renderer.render(view, L.layout(view, {tierOf: M.displayTier}),
    {roleInfo: M.roleInfo, semanticZoom: true, selected});
  flush();
}
function zoomBy(factor) { renderer.zoomBy(factor); flush(); }
"""
        source = modules + "\n" + harness + "\ntry {\n" + assertions + "\n} finally { renderer.destroy(); }\n"
        result = subprocess.run(
            [NODE, "-"], input=source, cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_zoom_events_reveal_four_tiers_without_moving_existing_nodes(self):
        self.run_renderer(r"""
render(viewOf(fixture()));
const firstPosition = {...cy.getElementById('tier1').position()};
const tiers = [lastInfo.tier], nodeCounts = [lastInfo.tierNodeCount];
for (const factor of [1.4, 1.5, 1.5]) {
  zoomBy(factor);
  tiers.push(lastInfo.tier); nodeCounts.push(lastInfo.tierNodeCount);
  assert.deepEqual(cy.getElementById('tier1').position(), firstPosition);
}
assert.deepEqual(tiers, [1, 2, 3, 4]);
assert.deepEqual(nodeCounts, [1, 2, 3, 4]);
assert.equal(lastInfo.totalLinks, 3);
assert.equal(lastInfo.tierLinkCount, 3);
zoomBy(1 / 3.15);
assert.equal(lastInfo.tier, 1);
assert.deepEqual(cy.nodes().map(node => node.id()), ['tier1']);
assert.deepEqual(cy.getElementById('tier1').position(), firstPosition);
""")

    def test_selected_detail_is_pinned_on_zoom_out_but_document_filter_remains_authoritative(self):
        self.run_renderer(r"""
const graph = fixture();
render(viewOf(graph));
zoomBy(3.15);
const zoom = cy.zoom(), pan = {...cy.pan()};
renderer.select('tier4', false); flush();
const position = {...cy.getElementById('tier4').position()};
assert.ok(cy.getElementById('tier4').selected());
assert.equal(cy.zoom(), zoom); assert.deepEqual(cy.pan(), pan);
zoomBy(1 / 3.15);
assert.equal(lastInfo.tier, 1);
assert.deepEqual(cy.nodes().map(node => node.id()).sort(), ['tier1', 'tier4']);
assert.deepEqual(cy.getElementById('tier4').position(), position);
assert.ok(cy.getElementById('tier4').selected());
assert.equal(lastInfo.tierNodeCount, 2);
render(viewOf(graph, {documentId: 'document-a'}), 'tier4');
assert.equal(lastInfo.totalNodes, 3);
assert.equal(cy.getElementById('tier4').length, 0);
assert.ok(lastInfo.nodes.every(node => node.id !== 'tier4'));
assert.ok(lastInfo.links.every(edge => edge.source !== 'tier4' && edge.target !== 'tier4'));
""")

    def test_search_keeps_fourth_tier_match_and_one_hop_neighbor_at_initial_zoom(self):
        self.run_renderer(r"""
render(viewOf(fixture(), {search: 'tier4'}));
assert.equal(lastInfo.tier, 4);
assert.equal(lastInfo.tierNodeCount, 2);
assert.equal(lastInfo.totalNodes, 2);
assert.deepEqual(cy.nodes().map(node => node.id()).sort(), ['tier3','tier4']);
assert.equal(lastInfo.links.length, 1);
assert.equal(relativeZoom, 1);
zoomBy(0.5);
assert.equal(lastInfo.tier, 4);
assert.deepEqual(cy.nodes().map(node => node.id()).sort(), ['tier3','tier4']);
""")

    def test_empty_search_clears_visible_notification_counts_selection_and_camera(self):
        self.run_renderer(r"""
const graph = fixture();
render(viewOf(graph)); zoomBy(3.15); renderer.select('tier4', false); flush();
assert.equal(lastInfo.totalNodes, 4);
const previousNotifications = notifications.length;
render(viewOf(graph, {search: 'missing-search-result'}), 'tier4');
assert.ok(notifications.length > previousNotifications, 'Empty scenes must notify the controller');
assert.deepEqual(lastInfo.nodes, []); assert.deepEqual(lastInfo.links, []);
assert.equal(lastInfo.totalNodes, 0); assert.equal(lastInfo.totalLinks, 0);
assert.equal(lastInfo.tierNodeCount, 0); assert.equal(lastInfo.tierLinkCount, 0);
assert.equal(lastInfo.tier, 1);
assert.equal(cy.nodes().length, 0); assert.equal(cy.edges().length, 0);
assert.equal(cy.elements(':selected').length, 0);
assert.equal(cy.zoom(), 1); assert.equal(relativeZoom, 1);
assert.deepEqual(cy.pan(), {x: 0, y: 0});
""")

    def test_large_course_starts_readably_and_whole_map_overview_is_separate_from_reset(self):
        self.run_renderer(r"""
const graph = M.normalize({nodes: Array.from({length: 60}, (_, index) =>
  concept('core-' + String(index).padStart(2, '0'), 1)), links: []});
const view = viewOf(graph), layout = L.layout(view, {tierOf: M.displayTier});
render(view);
assert.ok(cy.zoom() >= 0.6, 'Initial world scale must keep labels readable');
assert.equal(relativeZoom, 1); assert.equal(lastInfo.tier, 1);
assert.ok(parseFloat(cy.nodes().first().renderedStyle('font-size')) >= 10.79);
const initialZoom = cy.zoom();
renderer.fit(); flush();
assert.ok(cy.zoom() < initialZoom, 'Whole-map overview must fit a large course independently');
assert.ok(relativeZoom < 1); assert.equal(lastInfo.tier, 1);
assert.equal(cy.nodes().length, 60);
for (const node of view.nodes) {
  const p = layout.positions[node.id], size = layout.nodeSizes[node.id], z = cy.zoom(), pan = cy.pan();
  assert.ok(p.x * z + pan.x >= 49.99);
  assert.ok(p.y * z + pan.y >= 49.99);
  assert.ok((p.x + size.width) * z + pan.x <= 1215.01);
  assert.ok((p.y + size.height) * z + pan.y <= 556.01);
}
renderer.resetZoom(); flush();
assert.equal(cy.zoom(), initialZoom); assert.equal(relativeZoom, 1);
assert.equal(lastInfo.tier, 1);
""")


if __name__ == "__main__":
    unittest.main()
