"""Check the stable course map with explicitly simulated graph fixtures."""
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
LAYOUT = ROOT / "frontend/static/js/graph_map_layout.js"
BENCHMARK = ROOT / "scripts/benchmark_graph_map_layout.js"


@unittest.skipUnless(NODE, "Node.js is required for course map checks")
class GraphMapLayoutTests(unittest.TestCase):
    def js(self, code):
        prelude = "const assert=require('node:assert/strict');const L=require(" + repr(str(LAYOUT)) + ");\n"
        prelude += "const B=require(" + repr(str(BENCHMARK)) + ");\n"
        prelude += """
function draw(view){return L.layout(view,{tierOf:node=>node.tier||node.displayTier});}
function verify(view,drawing){
  assert.equal(Object.keys(drawing.positions).length,view.nodes.length);
  assert.equal(Object.keys(drawing.nodeSizes).length,view.nodes.length);
  for(const node of view.nodes){
    const p=drawing.positions[node.id],s=drawing.nodeSizes[node.id];
    assert.ok(Number.isFinite(p.x)&&Number.isFinite(p.y)&&p.x>=0&&p.y>=0);
    assert.ok(p.x+s.width<=drawing.width&&p.y+s.height<=drawing.height);
  }
  assert.equal(B.inspect(view,drawing).overlaps_with_16px_gap,0);
}
function freeze(value){if(value&&typeof value==='object'){Object.freeze(value);Object.values(value).forEach(freeze);}}
"""
        result = subprocess.run([NODE, "-e", prelude + code], capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_tier_sizes_are_explicit_and_every_rectangle_has_the_requested_gap(self):
        self.js("""
const view={nodes:[{id:'a',tier:1},{id:'b',tier:2},{id:'c',tier:3},{id:'d',tier:4}],links:[
 {source:'a',target:'b'},{source:'b',target:'c'},{source:'c',target:'d'}]};
const drawing=draw(view);
assert.deepEqual(Object.values(drawing.nodeSizes),[
 {width:240,height:76,fontSize:18},{width:204,height:64,fontSize:16},
 {width:164,height:52,fontSize:14},{width:132,height:42,fontSize:12}]);
assert.deepEqual(drawing.groups,[]);
verify(view,drawing);
""")

    def test_core_anchors_use_the_fixed_grid_and_degree_order(self):
        self.js("""
const view={nodes:[{id:'a',tier:1},{id:'b',tier:1},{id:'c',tier:1},{id:'d',tier:4},{id:'e',tier:4}],
 links:[{source:'a',target:'b'},{source:'b',target:'c'},{source:'b',target:'d'},{source:'b',target:'e'}]};
const drawing=draw(view),p=drawing.positions;
assert.deepEqual(drawing.anchorIds,['b','a','c']);
assert.equal(p.a.x-p.b.x,320);
assert.equal(p.c.x-p.a.x,320);
assert.equal(p.a.y,p.b.y);
assert.equal(p.c.y,p.b.y);
verify(view,drawing);
""")

    def test_no_tier_one_and_disconnected_detail_components_do_not_drop_nodes(self):
        self.js("""
const view={nodes:[{id:'a',tier:2},{id:'b',tier:3},{id:'x',tier:4},{id:'y',tier:4},{id:'isolated',tier:3}],
 links:[{source:'a',target:'b'},{source:'x',target:'y'}]};
const drawing=draw(view);
assert.equal(drawing.components.length,3);
assert.ok(drawing.anchorIds.includes('a'));
assert.ok(drawing.anchorIds.includes('isolated'));
assert.ok(drawing.anchorIds.some(id=>id==='x'||id==='y'));
for(const part of drawing.components)assert.ok(part.anchorIds.length>=1);
assert.deepEqual(drawing.components.flatMap(part=>part.nodeIds).sort(),view.nodes.map(node=>node.id).sort());
verify(view,drawing);
""")

    def test_layout_is_stable_under_input_reordering_and_ignores_zoom(self):
        self.js("""
const view=B.syntheticCourse(500),before=JSON.stringify(view);
freeze(view);
const first=draw(view),second=draw({...view,nodes:view.nodes.slice().reverse(),links:view.links.slice().reverse()});
assert.deepEqual(first.positions,second.positions);
assert.deepEqual(first.nodeSizes,second.nodeSizes);
assert.deepEqual(first.components,second.components);
const third=L.layout(view,{tierOf:node=>node.displayTier,zoom:4});
assert.deepEqual(first.positions,third.positions);
assert.equal(JSON.stringify(view),before);
verify(view,first);
""")

    def test_same_names_zero_and_prototype_like_ids_remain_distinct(self):
        self.js("""
const view={nodes:[{id:0,name:'Same',tier:1},{id:'__proto__',name:'Same',tier:2},{id:'constructor',name:'Same',tier:3},
 {id:'doc2:same',name:'Same',tier:4}],links:[{source:0,target:'__proto__'},{source:'constructor',target:'doc2:same'}]};
const drawing=draw(view);
assert.equal(Object.getPrototypeOf(drawing.positions),null);
assert.equal(Object.getPrototypeOf(drawing.nodeSizes),null);
assert.ok(drawing.positions['0']);assert.ok(drawing.positions['__proto__']);assert.ok(drawing.positions.constructor);
assert.equal(new Set(Object.values(drawing.positions).map(p=>p.x+':'+p.y)).size,4);
verify(view,drawing);
""")

    def test_spatial_queries_are_exact_at_cell_boundaries_and_negative_coordinates(self):
        self.js("""
const items=[{id:'negative',x:-40,y:-40,width:30,height:30},
 {id:'boundary',x:310,y:310,width:40,height:40},
 {id:'far',x:640,y:0,width:50,height:50},
 {id:'large',x:900,y:900,width:700,height:700}];
const index=L.spatialIndex(items,320);
const overlap=(a,b)=>a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y;
for(const box of [{x:-50,y:-50,width:100,height:100},{x:320,y:320,width:1,height:1},
 {x:350,y:310,width:10,height:10},{x:0,y:0,width:2000,height:2000},{x:1550,y:1550,width:100,height:100}])
 assert.deepEqual(index.query(box).sort(),items.filter(item=>overlap(box,item)).map(item=>item.id).sort());
assert.deepEqual(index.query({x:350,y:310,width:10,height:10}),[]);
assert.deepEqual(index.query({x:0,y:0,width:0,height:10}),[]);
assert.deepEqual(index.query({x:NaN,y:0,width:10,height:10}),[]);
""")

    def test_spatial_index_snapshots_input_rectangles_and_deduplicates_cell_hits(self):
        self.js("""
const item={id:'__proto__',x:1,y:1,width:700,height:700},items=[item];
const index=L.spatialIndex(items);
item.x=5000;items.push({id:'new',x:1,y:1,width:10,height:10});
assert.deepEqual(index.query({x:0,y:0,width:800,height:800}),['__proto__']);
assert.deepEqual(index.query({x:5000,y:0,width:800,height:800}),[]);
""")

    def test_simulated_5000_node_10000_edge_course_is_complete_and_collision_free(self):
        self.js("""
const view=B.syntheticCourse(5000),before=JSON.stringify(view),start=performance.now(),drawing=draw(view);
const elapsed=performance.now()-start;
assert.equal(view.simulated,true);assert.equal(view.nodes.length,5000);assert.equal(view.links.length,10000);
assert.equal(drawing.components.length,50);
verify(view,drawing);
assert.equal(JSON.stringify(view),before);
assert.ok(elapsed<2000,'synthetic 5000-node layout exceeded the 2s target: '+elapsed+'ms');
""")

    def test_single_dense_branch_uses_progressive_search_without_collisions(self):
        self.js("""
const nodes=[{id:'anchor',tier:1}],links=[];
for(let i=0;i<2000;i++){nodes.push({id:'leaf'+i,tier:2});links.push({source:'anchor',target:'leaf'+i});}
const view={nodes,links},start=performance.now(),drawing=draw(view);
assert.deepEqual(drawing.anchorIds,['anchor']);
verify(view,drawing);
assert.ok(performance.now()-start<2000);
""")

    def test_all_isolated_or_equal_tier_nodes_and_empty_input_are_supported(self):
        self.js("""
const view={nodes:Array.from({length:500},(_,i)=>({id:'isolated'+i,tier:4})),links:[]},drawing=draw(view);
assert.equal(drawing.anchorIds.length,500);
assert.equal(drawing.components.length,500);
verify(view,drawing);
const empty=L.layout({nodes:[],links:[]});
assert.deepEqual(Object.keys(empty.positions),[]);assert.deepEqual(empty.components,[]);
assert.deepEqual(L.spatialIndex([]).query({x:0,y:0,width:100,height:100}),[]);
""")

    def test_browser_global_umd_and_default_sizes_work_without_dependencies(self):
        self.js("""
const fs=require('node:fs'),vm=require('node:vm'),context={};
vm.createContext(context);vm.runInContext(fs.readFileSync(require.resolve(""" + repr(str(LAYOUT)) + """),'utf8'),context);
assert.equal(typeof context.CourseGraphMapLayout.layout,'function');
assert.equal(typeof context.CourseGraphMapLayout.spatialIndex,'function');
const result=context.CourseGraphMapLayout.layout({nodes:[{id:'a'}],links:[]});
assert.equal(result.nodeSizes.a.width,240);assert.equal(result.nodeSizes.a.fontSize,18);
""")


if __name__ == "__main__":
    unittest.main()
