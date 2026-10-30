"""Exercise semantic and spatial filtering together with synthetic graphs."""
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
JS = ROOT / "frontend/static/js"


@unittest.skipUnless(NODE, "Node.js is required for graph viewport checks")
class GraphViewportSubsetTests(unittest.TestCase):
    def js(self, code):
        prelude = "const assert=require('node:assert/strict');\n"
        for name, filename in (("M", "knowledge_graph_model.js"), ("L", "graph_map_layout.js"), ("C", "graph_canvas.js")):
            prelude += "const " + name + "=require(" + repr(str(JS / filename)) + ");\n"
        prelude += """
function concept(id,tier=1,extra={}) {
 return {id,name:id,type:'knowledge',importance_level:tier===1||tier===3?1:2,
  suggested_teaching_role:tier>=3?'detail':'learning',review_status:'pending',role_review_status:'pending',...extra};
}
function build(raw,positions,options={}) {
 const graph=M.normalize(raw),view=M.selectView(graph,{mode:'overview',showDetails:true,...options});
 const layout=L.layout(view,{tierOf:M.displayTier});
 if(positions)layout.positions=positions;
 const model=C.scene(view,layout,M.roleInfo),originals=new Map(view.nodes.map(node=>[node.id,node]));
 for(const node of model.nodes)node.displayTier=M.displayTier(originals.get(node.id));
 model.nodeById=new Map(model.nodes.map(node=>[node.id,node]));
 model.incident=new Map(model.nodes.map(node=>[node.id,[]]));
 for(const edge of model.edges){model.incident.get(edge.source).push(edge);model.incident.get(edge.target).push(edge);}
 const spatial=L.spatialIndex(model.nodes.map(node=>({id:node.id,x:node.position.x-node.width/2,
  y:node.position.y-node.height/2,width:node.width,height:node.height})));
 return {graph,view,layout,model,spatial};
}
function fixture() {
 return build({nodes:[concept('a',1),concept('b',2),concept('c',3),concept('d',4),concept('outside',4)],links:[
 {id:'ab',source:'a',target:'b',relation:'DRIVES'},
 {id:'bc',source:'b',target:'c',relation:'REQUIRES'},
 {id:'cd',source:'c',target:'d',relation:'PART_OF'},
 {id:'ao',source:'a',target:'outside',relation:'RELATED_TO'}]},
 {a:{x:0,y:0},b:{x:300,y:0},c:{x:0,y:150},d:{x:300,y:150},outside:{x:20000,y:20000}});
}
const viewport={x:-10,y:-10,width:600,height:300};
function freeze(value){if(value&&typeof value==='object'){Object.freeze(value);Object.values(value).forEach(freeze);}}
"""
        result = subprocess.run([NODE, "-e", prelude + code], capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_spatial_and_tier_filters_intersect_without_moving_scene_positions(self):
        self.js("""
const {model,spatial}=fixture(),before=JSON.stringify(model);
const expected=[['a'],['a','b'],['a','b','c'],['a','b','c','d']],edgeCounts=[0,1,2,3];
for(let tier=1;tier<=4;tier++){
 const result=C.viewportSubset(model,spatial,tier,viewport);
 assert.deepEqual(result.nodes.map(node=>node.id).sort(),expected[tier-1]);
 assert.equal(result.edges.length,edgeCounts[tier-1]);
 for(const node of result.nodes)assert.equal(node,model.nodeById.get(node.id));
}
assert.equal(JSON.stringify(model),before);
""")

    def test_selected_node_is_pinned_inside_or_outside_view_without_promoting_neighbours(self):
        self.js("""
const {model,spatial}=fixture();
const pinned=C.viewportSubset(model,spatial,1,viewport,'outside');
assert.deepEqual(pinned.nodes.map(node=>node.id).sort(),['a','outside']);
assert.deepEqual(pinned.edges.map(edge=>edge.recordId),['ao']);
assert.equal(pinned.nodes.find(node=>node.id==='outside').displayTier,4);
const inside=C.viewportSubset(model,spatial,1,viewport,'c');
assert.deepEqual(inside.nodes.map(node=>node.id).sort(),['a','c']);
assert.deepEqual(inside.edges,[]);
assert.deepEqual(C.viewportSubset(model,spatial,1,viewport,'unknown').nodes.map(node=>node.id),['a']);
""")

    def test_edges_are_original_records_and_runtime_needs_never_become_prerequisites(self):
        self.js("""
const {model,spatial}=fixture(),originals=new Map(model.edges.map(edge=>[edge.id,edge]));
const result=C.viewportSubset(model,spatial,4,viewport),ids=new Set(result.nodes.map(node=>node.id));
assert.deepEqual(result.edges.map(edge=>edge.recordId).sort(),['ab','bc','cd']);
for(const edge of result.edges){assert.equal(edge,originals.get(edge.id));assert.ok(ids.has(edge.source)&&ids.has(edge.target));assert.equal(edge.approved,false);}
assert.equal(result.edges.find(edge=>edge.recordId==='bc').label,'运行需要');
assert.equal(result.edges.find(edge=>edge.recordId==='cd').source,'c');
assert.equal(result.edges.find(edge=>edge.recordId==='cd').target,'d');
assert.equal(result.edges.find(edge=>edge.recordId==='cd').label,'组成部分');
assert.ok(!result.edges.some(edge=>edge.label==='先修'));
""")

    def test_parallel_relations_survive_and_incident_index_deduplicates_endpoint_hits(self):
        self.js("""
const data=build({nodes:[concept('a'),concept('b')],links:[
 {id:'one',source:'a',target:'b',relation:'DRIVES'},
 {id:'two',source:'a',target:'b',relation:'CONTROLS'},
 {id:'three',source:'b',target:'a',relation:'CONNECTS_TO'}]}, {a:{x:0,y:0},b:{x:300,y:0}});
const indexed=C.viewportSubset(data.model,data.spatial,4,viewport);
const fallback=C.viewportSubset({...data.model,nodeById:undefined,incident:undefined},data.spatial,4,viewport);
assert.deepEqual(indexed.edges.map(edge=>edge.recordId).sort(),['one','three','two']);
assert.deepEqual(indexed.edges.map(edge=>edge.id).sort(),fallback.edges.map(edge=>edge.id).sort());
assert.equal(indexed.edges.length,3);
""")

    def test_document_filter_is_authoritative_even_for_selected_other_document(self):
        self.js("""
const data=build({documents:[{document_id:'one'},{document_id:'two'}],nodes:[
 concept('one',1,{sources:[{document_id:'one',page_id:'one:1',page_no:1}]}),
 concept('two',1,{sources:[{document_id:'two',page_id:'two:1',page_no:1}]})],links:[
 {source:'one',target:'two',relation:'RELATED_TO'}]},undefined,{documentId:'one'});
const result=C.viewportSubset(data.model,data.spatial,4,{x:-100000,y:-100000,width:200000,height:200000},'two');
assert.deepEqual(result.nodes.map(node=>node.id),['one']);assert.deepEqual(result.edges,[]);
""")

    def test_pending_filter_and_special_ids_survive_viewport_culling(self):
        self.js("""
const raw={nodes:[concept(0),concept('__proto__',4),concept('constructor',2)],links:[
 {id:'pending',source:0,target:'__proto__',relation:'PART_OF',review_status:'pending'},
 {id:'approved',source:0,target:'constructor',relation:'CONNECTS_TO',review_status:'approved'}]};
const positions={'0':{x:0,y:0},['__proto__']:{x:20000,y:20000},constructor:{x:300,y:0}};
const all=build(raw,positions),confirmed=build(raw,positions,{showPending:false});
const selected=C.viewportSubset(all.model,all.spatial,1,viewport,'__proto__');
assert.deepEqual(selected.nodes.map(node=>node.id).sort(),['0','__proto__']);
assert.deepEqual(selected.edges.map(edge=>edge.recordId),['pending']);
const filtered=C.viewportSubset(confirmed.model,confirmed.spatial,4,viewport,'__proto__');
assert.deepEqual(filtered.edges.map(edge=>edge.recordId),['approved']);
assert.equal(filtered.edges[0].approved,true);
assert.deepEqual(C.viewportSubset(all.model,all.spatial,1,{x:10000,y:10000,width:10,height:10},'0').nodes.map(node=>node.id),['0']);
""")

    def test_unknown_spatial_ids_and_empty_viewport_cannot_add_phantom_nodes(self):
        self.js("""
const {model,spatial}=fixture();
const augmented={query:rect=>[...spatial.query(rect),'ghost']};
assert.deepEqual(C.viewportSubset(model,augmented,1,viewport).nodes.map(node=>node.id),['a']);
const empty=C.viewportSubset(model,spatial,4,{x:10000,y:10000,width:10,height:10});
assert.deepEqual(empty,{nodes:[],edges:[]});
""")

    def test_5000_nodes_10000_edges_clip_to_a_small_local_view_without_global_edge_scan(self):
        generator = ROOT / "scripts/benchmark_graph_map_layout.js"
        self.js("const generated=require(" + repr(str(generator)) + ").syntheticCourse(5000);\n" + """
const raw={simulation:{real_teaching_result:false},documents:Array.from({length:50},(_,i)=>({document_id:'simulation-doc'+i})),
 nodes:generated.nodes.map(node=>({...node,type:'knowledge',suggested_teaching_role:node.displayTier>=3?'detail':'learning',
 importance_level:node.displayTier%2===1?1:2})),links:generated.links};
assert.equal(raw.simulation.real_teaching_result,false);assert.equal(raw.documents.length,50);
const before=JSON.stringify(raw),{view,model,spatial}=build(raw);
assert.equal(view.nodes.length,5000);assert.equal(view.links.length,10000);
const first=model.nodes.find(node=>node.displayTier===1),rectangle={x:first.position.x-350,y:first.position.y-250,width:700,height:500};
const originals=new Map(model.edges.map(edge=>[edge.id,edge]));
// The optimized path must use incident lists rather than scan all 10,000 edges.
model.edges=new Proxy(model.edges,{get(target,property,receiver){if(property===Symbol.iterator)throw Error('full edge scan');return Reflect.get(target,property,receiver);}});
let previous=new Set();
for(let tier=1;tier<=4;tier++){
 const subset=C.viewportSubset(model,spatial,tier,rectangle),ids=new Set(subset.nodes.map(node=>node.id));
 assert.ok(subset.nodes.length>0&&subset.nodes.length<500);
 for(const id of previous)assert.ok(ids.has(id));
 for(const node of subset.nodes){assert.ok(node.displayTier<=tier);assert.equal(node,model.nodeById.get(node.id));}
 for(const edge of subset.edges){assert.equal(edge,originals.get(edge.id));assert.ok(ids.has(edge.source)&&ids.has(edge.target));}
 previous=ids;
}
assert.equal(JSON.stringify(raw),before);
""")


if __name__ == "__main__":
    unittest.main()
