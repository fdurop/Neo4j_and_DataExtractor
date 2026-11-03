"""Execute layout contracts in Node without browsers, providers or databases."""
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
MODEL = ROOT / "frontend/static/js/knowledge_graph_model.js"


@unittest.skipUnless(NODE, "Node.js is required for graph layout checks")
class GraphLayoutTests(unittest.TestCase):
    def js(self, code):
        prelude = "const assert=require('node:assert/strict');const M=require(" + repr(str(MODEL)) + ");\n"
        prelude += """
function graphView(names, pairs, mode='hierarchy') {
  const raw={nodes:names.map(id=>({id,name:id,type:'knowledge'})),
    links:pairs.map(([source,target])=>({source,target,relation:mode==='prerequisites'?'PREREQUISITE_OF':'HAS_PART'}))};
  return M.selectView(M.normalize(raw),{mode,showDetails:true});
}
function overlap(a,b) {return a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y;}
function checkBounds(view,drawing) {
  assert.equal(Object.keys(drawing.positions).length,view.nodes.length);
  const rectangles=view.nodes.map(node=>({...drawing.positions[node.id],width:drawing.nodeWidth,height:drawing.nodeHeight,id:node.id}));
  for(const rect of rectangles) {
    assert.ok(Number.isFinite(rect.x)&&Number.isFinite(rect.y));
    assert.ok(rect.x>=0&&rect.y>=0&&rect.x+rect.width<=drawing.width&&rect.y+rect.height<=drawing.height);
  }
  for(let i=0;i<rectangles.length;i++)for(let j=i+1;j<rectangles.length;j++)assert.equal(overlap(rectangles[i],rectangles[j]),false,rectangles[i].id+' overlaps '+rectangles[j].id);
  for(let i=0;i<drawing.components.length;i++)for(let j=i+1;j<drawing.components.length;j++)assert.equal(overlap(drawing.components[i],drawing.components[j]),false);
  for(const part of drawing.components)for(const id of part.nodeIds){const p=drawing.positions[id];assert.ok(p.x>=part.x&&p.y>=part.y&&p.x+drawing.nodeWidth<=part.x+part.width&&p.y+drawing.nodeHeight<=part.y+part.height);}
}
function sameSpanCrossings(view, drawing) {
  let count=0;
  for(let i=0;i<view.links.length;i++)for(let j=i+1;j<view.links.length;j++){
    const a=view.links[i],b=view.links[j],as=a.displaySource||a.source,at=a.displayTarget||a.target,bs=b.displaySource||b.source,bt=b.displayTarget||b.target;
    if(as===bs||at===bt)continue;
    const p=drawing.positions[as],q=drawing.positions[at],r=drawing.positions[bs],s=drawing.positions[bt];
    if(p.x===r.x&&q.x===s.x&&(p.y-r.y)*(q.y-s.y)<0)count++;
  }
  return count;
}
"""
        result = subprocess.run([NODE, "-e", prelude + code], capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_disconnected_components_are_separate_blocks_without_node_overlap(self):
        self.js("""
const view=graphView(['a','b','c','d','e','f','g'],[['a','b'],['a','c'],['d','e'],['e','f'],['e','g']]);
const drawing=M.layout(view);
assert.equal(drawing.components.length,2);
assert.deepEqual(drawing.components.map(c=>c.nodeIds.length).sort((a,b)=>a-b),[3,4]);
assert.equal(drawing.groups.length,0);
checkBounds(view,drawing);
for(const edge of view.links)assert.ok(drawing.positions[edge.displaySource].x<drawing.positions[edge.displayTarget].x);
""")

    def test_barycentric_order_removes_an_actual_alphabetical_crossing(self):
        self.js("""
// Alphabetical columns a,b / y,z cross a->z with b->y.
const view=graphView(['root','a','b','y','z'],[['root','a'],['root','b'],['a','z'],['b','y']]);
const alphabetical={positions:{root:{x:0,y:0},a:{x:1,y:0},b:{x:1,y:1},y:{x:2,y:0},z:{x:2,y:1}}};
assert.equal(sameSpanCrossings(view,alphabetical),1);
const drawing=M.layout(view);
assert.equal(sameSpanCrossings(view,drawing),0);
assert.equal(drawing.components[0].crossings,0);
checkBounds(view,drawing);
""")

    def test_longest_path_layers_keep_multiple_parents_and_skip_edges_forward(self):
        self.js("""
const view=graphView(['r','a','b','c','d'],[['r','a'],['a','b'],['b','c'],['r','c'],['a','d'],['d','c']],'prerequisites');
const drawing=M.layout(view);
assert.equal(drawing.cycle,false);
assert.equal(drawing.components[0].levelCount,4);
for(const edge of view.links)assert.ok(drawing.positions[edge.source].x<drawing.positions[edge.target].x);
checkBounds(view,drawing);
""")

    def test_cycle_warns_and_uses_only_its_component_grid(self):
        self.js("""
const view=graphView(['a','b','c','x','y'],[['a','b'],['b','c'],['c','a'],['x','y']]);
const drawing=M.layout(view);
assert.equal(drawing.cycle,true);
assert.ok(drawing.warnings.some(w=>w.includes('循环')));
assert.equal(drawing.components.filter(part=>part.cycle).length,1);
const cyclic=drawing.components.find(part=>part.cycle);
assert.equal(cyclic.levelCount,null);
assert.equal(cyclic.crossings,null);
assert.ok(drawing.positions.x.x<drawing.positions.y.x);
checkBounds(view,drawing);
""")

    def test_component_packing_avoids_a_single_tall_column_for_unrelated_branches(self):
        self.js("""
const names=[],pairs=[];
for(const [index,size] of [12,7,6,4,4].entries()){
  const root='component'+index+'-root';names.push(root);
  for(let child=1;child<size;child++){const id='component'+index+'-child'+child;names.push(id);pairs.push([root,id]);}
}
const view=graphView(names,pairs),drawing=M.layout(view);
assert.equal(drawing.components.length,5);
assert.ok(drawing.height<2000,'component packing should not retain the old 28-row global leaf column');
assert.ok(new Set(drawing.components.map(part=>part.x)).size>1,'use horizontal space beside the tallest branch');
checkBounds(view,drawing);
""")

    def test_overview_retains_document_and_page_source_grouping(self):
        self.js("""
const raw={documents:[{document_id:'first',file_name:'First notes'},{document_id:'second',file_name:'Second notes'}],
 concepts:[{id:'a',canonical_name:'A',sources:[{document_id:'second',page_no:3}]},
 {id:'b',canonical_name:'B',sources:[{document_id:'first',page_no:7}]},
 {id:'c',canonical_name:'C',sources:[{document_id:'first',page_no:2}]}],relationships:[]};
const view=M.selectView(M.normalize(raw),{mode:'overview',showDetails:true}),drawing=M.layout(view);
assert.deepEqual(drawing.groups.map(group=>group.documentId),['first','second']);
assert.deepEqual(drawing.groups[0].nodeIds,['c','b']);
assert.equal(drawing.groups[0].label,'First notes');
assert.deepEqual(drawing.components,[]);
checkBounds(view,drawing);
""")

    def test_layout_is_deterministic_and_does_not_mutate_inputs(self):
        self.js("""
const view=graphView(['root','a','b','y','z','other','leaf'],[['root','a'],['root','b'],['a','z'],['b','y'],['other','leaf']]);
const frozen=JSON.stringify(view);
function freeze(value){if(value&&typeof value==='object'){Object.freeze(value);Object.values(value).forEach(freeze);}}freeze(view);
const first=M.layout(view);
assert.equal(JSON.stringify(view),frozen);
const shuffled={...view,nodes:view.nodes.slice().reverse(),links:view.links.slice().reverse()};
const second=M.layout(shuffled);
assert.deepEqual(first.positions,second.positions);
assert.deepEqual(first.components,second.components);
""")

    def test_prototype_like_ids_and_display_direction_keep_contract(self):
        self.js("""
const raw={nodes:[{id:0,name:'Root',type:'knowledge'},{id:'__proto__',name:'Part',type:'knowledge'},
 {id:'constructor',name:'Subtype',type:'knowledge'}],links:[
 {source:'__proto__',target:0,relation:'PART_OF'},
 {source:'constructor',target:0,relation:'IS_A'}]};
const graph=M.normalize(raw),view=M.selectView(graph,{mode:'hierarchy',showDetails:true}),drawing=M.layout(view);
assert.equal(Object.getPrototypeOf(drawing.positions),null);
assert.ok(drawing.positions['0'].x<drawing.positions['__proto__'].x);
assert.ok(drawing.positions['0'].x<drawing.positions.constructor.x);
assert.equal(view.links[0].source,'__proto__');
checkBounds(view,drawing);
""")


if __name__ == "__main__":
    unittest.main()
