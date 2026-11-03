"use strict";

// Synthetic scale fixtures only. They are not extracted or validated lessons.
const fs = require("node:fs"), path = require("node:path"), {performance} = require("node:perf_hooks");
const L = require("../frontend/static/js/graph_map_layout.js");

function syntheticCourse(count) {
  const nodes = [], links = [], block = 100;
  for (let index = 0; index < count; index++) {
    const document = Math.floor(index / block), id = "simulation-doc" + document + ":node" + index;
    nodes.push({id, name: "模拟知识点" + index % 12, displayTier: index % 4 + 1,
      sources: [{document_id: "simulation-doc" + document, page_no: index % block + 1}]});
  }
  for (let index = 0; index < count; index++) {
    const start = Math.floor(index / block) * block, length = Math.min(block, count - start), local = index - start;
    for (const offset of [1, 7]) links.push({id: "simulation-edge" + index + ":" + offset,
      source: nodes[index].id, target: nodes[start + (local + offset) % length].id, relation: "RELATED_TO"});
  }
  return {nodes, links, simulated: true};
}

function inspect(view, drawing) {
  const boxes = view.nodes.map(node => ({id: node.id, ...drawing.positions[node.id], ...drawing.nodeSizes[node.id]}));
  const index = L.spatialIndex(boxes, 320);
  let overlaps = 0;
  for (const box of boxes) {
    // Inflate by 16 to verify the requested gap as well as actual overlap.
    const neighbours = index.query({x: box.x - 16, y: box.y - 16, width: box.width + 32, height: box.height + 32});
    overlaps += neighbours.filter(id => id !== box.id && id > box.id).length;
  }
  return {nodes_laid_out: Object.keys(drawing.positions).length, overlaps_with_16px_gap: overlaps,
    width: drawing.width, height: drawing.height, component_count: drawing.components.length, anchor_count: drawing.anchorIds.length};
}

function measure(count) {
  const view = syntheticCourse(count), started = performance.now();
  const drawing = L.layout(view, {tierOf: node => node.displayTier}), elapsed = performance.now() - started;
  return {nodes_input: view.nodes.length, edges_input: view.links.length, layout_ms: Number(elapsed.toFixed(3)), ...inspect(view, drawing)};
}

function run(output) {
  L.layout(syntheticCourse(200), {tierOf: node => node.displayTier});
  const results = [500, 2000, 5000].map(measure);
  const report = {author: "昕煜 冷", created_at: new Date().toISOString(), fixture_type: "simulated_course_scale",
    fixture_version: "segmented-course-v1", real_teaching_result: false,
    note: "模拟课程规模与布局性能测试，不代表真实教学成果或知识关系质量。",
    runtime: {node: process.version, platform: process.platform, architecture: process.arch},
    target_layout_ms: 2000, results,
    passed: results.every(row => row.nodes_input === row.nodes_laid_out && row.overlaps_with_16px_gap === 0 && row.layout_ms < 2000)};
  fs.mkdirSync(path.dirname(output), {recursive: true});
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + "\n", "utf8");
  return report;
}

module.exports = {syntheticCourse, inspect, measure, run};
if (require.main === module) {
  const output = path.resolve(process.argv[2] || path.join(__dirname, "../runtime/course_scale/2026-10-24/map_layout_benchmark.json"));
  const report = run(output);
  console.log(JSON.stringify({output, passed: report.passed, results: report.results}));
  if (!report.passed) process.exitCode = 1;
}
