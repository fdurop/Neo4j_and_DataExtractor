"""Offline graph viewport contracts using the shipped Cytoscape core."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "frontend" / "static" / "js"
NODE = shutil.which("node")
NS = {"svg": "http://www.w3.org/2000/svg"}


@unittest.skipUnless(NODE, "Node.js is required for the frontend contract tests")
class GraphCanvasTests(unittest.TestCase):
    def node_json(self, source):
        preamble = "\n".join((
            '"use strict";',
            "const M = require(%s);" % json.dumps(str(JS / "graph_canvas.js")),
            "const cytoscape = require(%s);" % json.dumps(str(JS / "vendor" / "cytoscape.min.js")),
            "function freeze(value) {",
            "  if (value && typeof value === 'object') {",
            "    Object.freeze(value); Object.values(value).forEach(freeze);",
            "  } return value;",
            "}",
            "function roleInfo(node) { return {key: 'learning', label: '学习知识点', confirmed: node.id === 'parent'}; }",
        ))
        result = subprocess.run(
            [NODE, "-"], input=preamble + "\n" + source,
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            self.fail("Node did not return JSON: %s\n%s" % (exc, result.stdout))

    def test_actual_styles_scale_world_font_and_node_width_with_zoom(self):
        result = self.node_json(r"""
const cy = cytoscape({headless: true, styleEnabled: true, layout: {name: 'preset'},
  style: M.styles(), elements: [
    {data: {id: 'a', label: '前置知识', width: 220, height: 58, textWidth: 192,
      fill: '#ffffff', border: '#a6b8b0'}, position: {x: 0, y: 0}},
    {data: {id: 'b', label: '学习目标', width: 220, height: 58, textWidth: 192,
      fill: '#ffffff', border: '#a6b8b0'}, position: {x: 300, y: 0}},
    {data: {id: 'e', source: 'a', target: 'b', relationLabel: '先修',
      shownLabel: '', lineStyle: 'dashed'}}
  ]});
try {
  const n = cy.getElementById('a'), e = cy.getElementById('e');
  const samples = [0.5, 1, 2].map(zoom => {
    cy.zoom(zoom);
    return {zoom: cy.zoom(), worldFont: parseFloat(n.style('font-size')),
      renderedFont: parseFloat(n.renderedStyle('font-size')),
      worldWidth: n.width(), renderedWidth: n.renderedWidth(),
      edgeWorldFont: parseFloat(e.style('font-size')),
      edgeRenderedFont: parseFloat(e.renderedStyle('font-size'))};
  });
  const ordinary = {nodeTextBackground: n.style('text-background-opacity'),
    edgeTextBackground: e.style('text-background-opacity'), edgeLabel: e.style('label'),
    edgeOutline: e.style('text-outline-width'), arrow: e.style('target-arrow-shape'),
    pendingLine: e.style('line-style')};
  e.addClass('active');
  const active = {label: e.style('label'), background: e.style('text-background-opacity')};
  e.removeClass('active'); e.data('shownLabel', '先修'); e.data('lineStyle', 'solid');
  const labelled = {label: e.style('label'), background: e.style('text-background-opacity'),
    approvedLine: e.style('line-style')};
  console.log(JSON.stringify({samples, ordinary, active, labelled}));
} finally { cy.destroy(); }
""")
        for sample in result["samples"]:
            zoom = sample["zoom"]
            self.assertEqual(sample["worldFont"], 14)
            self.assertAlmostEqual(sample["renderedFont"], 14 * zoom)
            self.assertEqual(sample["worldWidth"], 220)
            self.assertAlmostEqual(sample["renderedWidth"], 220 * zoom)
            self.assertEqual(sample["edgeWorldFont"], 11)
            self.assertAlmostEqual(sample["edgeRenderedFont"], 11 * zoom)
        self.assertEqual(float(result["ordinary"]["nodeTextBackground"]), 0)
        self.assertEqual(float(result["ordinary"]["edgeTextBackground"]), 0)
        self.assertEqual(float(result["ordinary"]["edgeOutline"].removesuffix("px")), 0)
        self.assertEqual(result["ordinary"]["edgeLabel"], "")
        self.assertEqual(result["ordinary"]["arrow"], "triangle")
        self.assertEqual(result["ordinary"]["pendingLine"], "dashed")
        self.assertEqual(result["active"]["label"], "先修")
        self.assertEqual(float(result["active"]["background"]), 0)
        self.assertEqual(result["labelled"]["label"], "先修")
        self.assertEqual(float(result["labelled"]["background"]), 0)
        self.assertEqual(result["labelled"]["approvedLine"], "solid")

    def fixture(self):
        return self.node_json(r"""
const view = freeze({mode: 'hierarchy', nodes: [
  {id: 'child', name: '目标知识', description: '后续学习目标'},
  {id: 'parent', name: '基础知识', description: '相关基础'},
  {id: 'extra', name: '延伸知识', description: '另一章内容'}
], links: [
  {id: 'pending', source: 'child', target: 'parent', displaySource: 'parent',
    displayTarget: 'child', relation: 'IS_A', reviewStatus: 'pending', description: '待核分类'},
  {id: 'approved', source: 'child', target: 'extra', relation: 'PART_OF',
    reviewStatus: 'approved', description: '已核包含'}
]});
const layout = freeze({nodeWidth: 220, nodeHeight: 58, positions: {
  child: {x: 100, y: 150}, parent: {x: -500, y: -300}, extra: {x: -700, y: 250}
}, groups: []});
const before = JSON.stringify({view, layout});
const model = M.scene(view, layout, roleInfo);
const modelBefore = JSON.stringify(model);
const plain = M.svgMarkup(model);
const labelled = M.svgMarkup(model, {showEdgeLabels: true});
const titled = M.svgMarkup(model, {title: '课程知识图谱', subtitle: '教师核对版'});
console.log(JSON.stringify({nodes: model.nodes, edges: model.edges,
  bounds: M.bounds(model.nodes, model.width, model.height),
  nodeWidth: model.width, nodeHeight: model.height, plain, labelled, titled,
  inputUnchanged: before === JSON.stringify({view, layout}),
  modelUnchanged: modelBefore === JSON.stringify(model)}));
""")

    def test_scene_preserves_display_direction_review_state_and_source_data(self):
        result = self.fixture()
        self.assertTrue(result["inputUnchanged"])
        self.assertTrue(result["modelUnchanged"])
        pending, approved = result["edges"]
        self.assertEqual((pending["source"], pending["target"]), ("parent", "child"))
        self.assertEqual(pending["recordId"], "pending")
        self.assertEqual(pending["label"], "分类")
        self.assertEqual(pending["description"], "待核分类")
        self.assertFalse(pending["approved"])
        self.assertEqual((approved["source"], approved["target"]), ("child", "extra"))
        self.assertEqual(approved["label"], "包含")
        self.assertTrue(approved["approved"])
        positions = {node["id"]: node["position"] for node in result["nodes"]}
        self.assertEqual(positions["parent"], {"x": -390, "y": -271})

    def test_svg_keeps_target_arrows_and_only_pending_edges_are_dashed(self):
        result = self.fixture()
        root = ET.fromstring(result["plain"]["markup"])
        lines = root.find("svg:g", NS).findall("svg:path", NS)
        self.assertEqual(len(lines), 2)
        nodes = {node["id"]: node["position"] for node in result["nodes"]}
        for edge, line in zip(result["edges"], lines):
            self.assertEqual(line.get("marker-end"), "url(#kg-arrow)")
            self.assertIsNone(line.get("marker-start"))
            self.assertEqual(line.get("stroke-dasharray"), None if edge["approved"] else "5 4")
            numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", line.get("d"), re.I)]
            start_x, start_y, end_x, end_y = numbers[0], numbers[1], numbers[-2], numbers[-1]
            source, target = nodes[edge["source"]], nodes[edge["target"]]
            directed_distance = ((end_x - start_x) * (target["x"] - source["x"])
                                 + (end_y - start_y) * (target["y"] - source["y"]))
            self.assertGreater(directed_distance, 0, "SVG path must run from display source to target")

    def test_edge_labels_are_opt_in_and_do_not_add_text_background_rectangles(self):
        result = self.fixture()
        plain = ET.fromstring(result["plain"]["markup"])
        labelled = ET.fromstring(result["labelled"]["markup"])
        self.assertEqual(plain.find("svg:g", NS).findall("svg:text", NS), [])
        self.assertEqual([text.text for text in labelled.find("svg:g", NS).findall("svg:text", NS)],
                         ["分类", "包含"])
        # One canvas background, one node body and one narrow role stripe per node.
        expected_rectangles = 1 + 2 * len(result["nodes"])
        self.assertEqual(len(plain.findall(".//svg:rect", NS)), expected_rectangles)
        self.assertEqual(len(labelled.findall(".//svg:rect", NS)), expected_rectangles)

    def test_svg_bounds_include_every_negative_and_positive_node(self):
        result = self.fixture()
        bounds = result["bounds"]
        self.assertEqual(bounds, {"x": -700, "y": -300, "width": 1020, "height": 608})
        for kind, expected_top in (("plain", 36), ("titled", 98)):
            with self.subTest(export=kind):
                exported = result[kind]
                root = ET.fromstring(exported["markup"])
                width, height = float(root.get("width")), float(root.get("height"))
                self.assertEqual(root.get("viewBox"), "0 0 %d %d" % (width, height))
                self.assertEqual(width, exported["width"])
                self.assertEqual(height, exported["height"])
                group = root.find("svg:g", NS)
                offset_x, offset_y = map(float, re.fullmatch(
                    r"translate\(([-\d.]+) ([-\d.]+)\)", group.get("transform")).groups())
                extents = []
                for node in result["nodes"]:
                    left = node["position"]["x"] - result["nodeWidth"] / 2 + offset_x
                    top = node["position"]["y"] - result["nodeHeight"] / 2 + offset_y
                    right, bottom = left + result["nodeWidth"], top + result["nodeHeight"]
                    self.assertGreaterEqual(left, 36)
                    self.assertGreaterEqual(top, expected_top)
                    self.assertLessEqual(right, width - 36)
                    self.assertLessEqual(bottom, height - 36)
                    extents.append((left, top, right, bottom))
                self.assertEqual(min(x[0] for x in extents), 36)
                self.assertEqual(min(x[1] for x in extents), expected_top)

    def test_names_descriptions_titles_and_relation_labels_cannot_inject_svg(self):
        result = self.node_json(r"""
const names = ['<svg onload="boom">', '&\'"<>', '知识🚀点'];
const description = '</text><script>descriptionAttack()</script><image href="https://invalid.test/x"/>';
const view = freeze({mode: 'prerequisite', nodes: names.map((name, i) =>
  ({id: 'n' + i, name, description})), links: [
    {id: 'e', source: 'n0', target: 'n1', reviewStatus: 'pending', description}
  ]});
const layout = freeze({positions: {n0: {x: 0, y: 0}, n1: {x: 300, y: 0},
  n2: {x: 600, y: 0}}, nodeWidth: 220, nodeHeight: 58});
const roleLabel = '<img onerror="roleAttack()">';
const before = JSON.stringify({view, layout});
const model = M.scene(view, layout, () => ({key: 'learning', label: roleLabel}));
model.edges[0].label = '</text><script>edgeAttack()</script>';
const exported = M.svgMarkup(model, {showEdgeLabels: true,
  title: '</text><script>titleAttack()</script>', subtitle: '<image onload="subtitleAttack()"/>'});
console.log(JSON.stringify({markup: exported.markup, names, roleLabel, description,
  sceneDescriptions: model.nodes.map(n => n.description), edgeDescription: model.edges[0].description,
  inputUnchanged: before === JSON.stringify({view, layout})}));
""")
        self.assertTrue(result["inputUnchanged"])
        self.assertEqual(result["sceneDescriptions"], [result["description"]] * 3)
        self.assertEqual(result["edgeDescription"], result["description"])
        root = ET.fromstring(result["markup"])
        allowed = {"svg", "rect", "defs", "marker", "path", "text", "g", "title"}
        for element in root.iter():
            self.assertIn(element.tag.rsplit("}", 1)[-1], allowed)
            for attribute in element.attrib:
                local_name = attribute.rsplit("}", 1)[-1].lower()
                self.assertFalse(local_name.startswith("on"), "No event-handler attribute may be injected")
                self.assertNotIn(local_name, {"href", "src", "style"})
        titles = [element.text for element in root.findall(".//svg:title", NS)]
        self.assertEqual(titles, [name + " · " + result["roleLabel"] for name in result["names"]])
        label_text = [element.text for element in root.findall(".//svg:text", NS)]
        self.assertIn('</text><script>titleAttack()</script>', label_text)
        self.assertIn('</text><script>edgeAttack()</script>', label_text)
        self.assertIn('&\'"<>', label_text)


if __name__ == "__main__":
    unittest.main()
