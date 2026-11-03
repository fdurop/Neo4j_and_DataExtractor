"""Verify semantic zoom against graph data without browsers or paid services."""
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
MODEL = ROOT / "frontend/static/js/knowledge_graph_model.js"


@unittest.skipUnless(NODE, "Node.js is required for semantic zoom checks")
class GraphSemanticZoomTests(unittest.TestCase):
    def js(self, code):
        prelude = "const assert=require('node:assert/strict');const M=require(" + repr(str(MODEL)) + ");\n"
        prelude += """
function concept(id, importance=1, role='learning', extra={}) {
  return {id,name:id,type:'knowledge',importance_level:importance,
    suggested_teaching_role:role,review_status:'pending',role_review_status:'pending',...extra};
}
function view(nodes, links=[], options={}) {
  return M.selectView(M.normalize({nodes,links}),{mode:'overview',showDetails:true,...options});
}
function freeze(value) {if(value&&typeof value==='object'){Object.freeze(value);Object.values(value).forEach(freeze);}}
"""
        result = subprocess.run([NODE, "-e", prelude + code], capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_four_tiers_and_cumulative_node_and_relation_counts(self):
        self.js("""
const v=view([concept('main'),concept('context',1,'context_dependent'),concept('support',2),
  concept('detail',1,'detail'),concept('fine',2,'detail'),concept('fine3',3,'detail')],
  [{source:'main',target:'support',relation:'REQUIRES'},
   {source:'detail',target:'main',relation:'PART_OF'},
   {source:'fine',target:'detail',relation:'CONNECTS_TO'}]);
assert.deepEqual(v.nodes.map(M.displayTier),[1,1,2,3,4,4]);
const p=M.tierProfile(v);
assert.deepEqual(p.counts,[2,1,1,2]);
assert.deepEqual(p.cumulative,[2,3,4,6]);
assert.deepEqual(p.linkCounts,[0,1,2,3]);
assert.equal(p.total,6);
assert.equal(p.minTier,1);
for(let tier=1;tier<=4;tier++){
  const visible=M.semanticView(v,tier);
  assert.equal(visible.nodes.length,p.cumulative[tier-1]);
  assert.equal(visible.links.length,p.linkCounts[tier-1]);
  assert.equal(visible.hiddenCount,6-visible.nodes.length);
}
""")

    def test_teacher_role_decision_overrides_suggestion_without_changing_status(self):
        self.js("""
const v=view([concept('confirmed-detail',1,'learning',{teaching_role:'detail',role_review_status:'approved'}),
  concept('confirmed-learning',1,'detail',{teaching_role:'learning',role_review_status:'approved'}),
  concept('rejected-suggestion',1,'detail',{role_review_status:'rejected'}),
  concept('uncertain',2,'pending')]);
assert.deepEqual(v.nodes.map(M.displayTier),[3,1,1,2]);
const before=JSON.stringify(v),p=M.tierProfile(v),s=M.semanticView(v,4);
assert.equal(p.byId['confirmed-detail'],3);
assert.equal(s.nodes[0].reviewStatus,'pending');
assert.equal(s.nodes[0].roleReviewStatus,'approved');
assert.equal(s.nodes[2].roleReviewStatus,'rejected');
assert.equal(JSON.stringify(v),before);
""")

    def test_zoom_boundaries_and_hysteresis_prevent_flicker(self):
        self.js("""
assert.deepEqual([1,1.249,1.25,1.799,1.8,2.599,2.6,10].map(z=>M.zoomTier(z)),[1,1,2,2,3,3,4,4]);
for(const [boundary,low,high] of [[1.25,1,2],[1.8,2,3],[2.6,3,4]]){
  assert.equal(M.zoomTier(boundary*1.079,low),low);
  assert.equal(M.zoomTier(boundary*1.081,low),high);
  assert.equal(M.zoomTier(boundary*0.921,high),high);
  assert.equal(M.zoomTier(boundary*0.919,high),low);
  assert.equal(M.zoomTier(boundary,low),low);
  assert.equal(M.zoomTier(boundary,high),high);
}
assert.equal(M.zoomTier(10,1),4);
assert.equal(M.zoomTier(1,4),1);
assert.equal(M.zoomTier('1.8'),3);
assert.equal(M.zoomTier(NaN),1);
assert.equal(M.zoomTier(Infinity,3),3);
assert.equal(M.zoomTier(0,2),2);
""")

    def test_overview_keeps_original_relation_types_and_directions(self):
        self.js("""
const relations=['PART_OF','IS_A','HAS_PART','REQUIRES','CONNECTS_TO','CONTROLS','DRIVES','FLOWS_TO','RELATED_TO','PREREQUISITE_OF'];
const v=view([concept('a'),concept('b')],relations.map((relation,i)=>({id:'e'+i,source:'a',target:'b',relation})));
assert.deepEqual(v.links.map(e=>e.relation),relations);
for(const e of v.links){assert.equal(e.source,'a');assert.equal(e.target,'b');assert.equal(e.displaySource,'a');assert.equal(e.displayTarget,'b');assert.equal(e.reviewStatus,'pending');}
const s=M.semanticView(v,1);
assert.deepEqual(s.links.map(e=>e.relation),relations);
assert.equal(s.links.filter(e=>e.relation==='PREREQUISITE_OF').length,1);
""")

    def test_management_and_material_edges_and_candidates_never_become_semantics(self):
        self.js("""
const management=['HAS_EVIDENCE','CONTAINS','BELONGS_TO','NEXT_PAGE','PREV_PAGE','PREVIOUS_PAGE',
 'COURSE_CONTAINS_DOCUMENT','DOCUMENT_CONTAINS_PAGE','PAGE_CONTAINS_CONTENT'];
const g=M.normalize({nodes:[concept('a'),concept('b'),{id:'material',name:'Image',type:'Image'}],
 links:[...management.map((relation,i)=>({id:'m'+i,source:'a',target:'b',relation})),
  {id:'image-edge',source:'a',target:'material',relation:'RELATED_TO'},
  {id:'real',source:'a',target:'b',relation:'HAS_PARAMETER'},
  {id:'derived',source:'a',target:'b',relation:'PART_OF',auto_generated:true}],
 hierarchy_candidates:[{id:'candidate',source:'a',target:'b',relation:'HAS_PART',review_status:'approved'}]});
assert.equal(g.hierarchyCandidates.length,2);
const v=M.selectView(g,{mode:'overview',showDetails:true});
assert.deepEqual(v.links.map(e=>e.id),['real']);
assert.deepEqual(M.semanticView(g,4).links.map(e=>e.id),['real']);
assert.equal(M.tierProfile(g).total,2);
""")

    def test_pending_edge_filter_survives_semantic_zoom(self):
        self.js("""
const rawLinks=[{id:'approved',source:'a',target:'b',relation:'RELATED_TO',review_status:'approved'},
 {id:'pending',source:'b',target:'c',relation:'CONNECTS_TO',review_status:'pending'},
 {id:'rejected',source:'c',target:'a',relation:'PART_OF',review_status:'rejected'}];
const nodes=[concept('a'),concept('b'),concept('c')];
const all=view(nodes,rawLinks),confirmed=view(nodes,rawLinks,{showPending:false});
assert.deepEqual(all.links.map(e=>e.id),['approved','pending']);
assert.deepEqual(M.semanticView(confirmed,4).links.map(e=>e.id),['approved']);
assert.equal(M.semanticView(confirmed,4).links[0].reviewStatus,'approved');
assert.equal(all.links[1].reviewStatus,'pending');
""")

    def test_only_edges_with_two_visible_endpoints_are_shown(self):
        self.js("""
const v=view([concept('a'),concept('b',2),concept('c',1,'detail'),concept('d',3,'detail')],
 [{id:'12',source:'a',target:'b',relation:'CONTROLS'},
  {id:'13',source:'a',target:'c',relation:'DRIVES'},
  {id:'34',source:'c',target:'d',relation:'PART_OF'}]);
assert.deepEqual(M.semanticView(v,1).links,[]);
assert.deepEqual(M.semanticView(v,2).links.map(e=>e.id),['12']);
assert.deepEqual(M.semanticView(v,3).links.map(e=>e.id),['12','13']);
for(let tier=1;tier<=4;tier++){
  const visible=M.semanticView(v,tier),ids=new Set(visible.nodes.map(n=>n.id));
  assert.ok(visible.links.every(e=>ids.has(e.source)&&ids.has(e.target)));
}
""")

    def test_selected_detail_can_be_pinned_without_promoting_neighbours(self):
        self.js("""
const v=view([concept('main'),concept('detail',3,'detail'),concept('neighbour',3,'detail')],
 [{source:'main',target:'detail',relation:'PART_OF'},{source:'detail',target:'neighbour',relation:'RELATED_TO'}]);
const pinned=M.semanticView(v,1,'detail');
assert.deepEqual(pinned.nodes.map(n=>n.id),['main','detail']);
assert.deepEqual(pinned.pinnedIds,['detail']);
assert.equal(pinned.hiddenCount,1);
assert.equal(pinned.links.length,1);
assert.equal(M.displayTier(pinned.nodes[1]),4);
assert.deepEqual(M.semanticView(v,1,'missing').nodes.map(n=>n.id),['main']);
assert.deepEqual(M.semanticView(v,4,'detail').pinnedIds,[]);
""")

    def test_document_search_and_importance_filters_stay_authoritative(self):
        self.js("""
const g=M.normalize({documents:[{document_id:'one'},{document_id:'two'}],concepts:[
 {id:'shared',canonical_name:'Shared',importance_level:1,suggested_teaching_role:'learning',sources:[{page_id:'one4',document_id:'one',page_no:4},{page_id:'two2',document_id:'two',page_no:2}]},
 {id:'one',canonical_name:'One',importance_level:2,suggested_teaching_role:'learning',sources:[{page_id:'one1',document_id:'one',page_no:1}]},
 {id:'two',canonical_name:'Two',importance_level:1,suggested_teaching_role:'detail',sources:[{page_id:'two3',document_id:'two',page_no:3}]}],relationships:[
 {source:'shared',target:'one',relation:'REQUIRES'},{source:'shared',target:'two',relation:'HAS_PART'}]});
const one=M.selectView(g,{mode:'overview',showDetails:true,documentId:'one'});
assert.deepEqual(one.nodes.map(n=>n.id),['shared','one']);
assert.deepEqual(M.semanticView(one,4,'two').nodes.map(n=>n.id),['shared','one']);
assert.equal(one.nodes[0].viewPrimarySource.page_no,4);
assert.equal(M.tierProfile(one).total,2);
const two=M.selectView(g,{mode:'overview',showDetails:true,documentId:'two',search:'Two'});
assert.deepEqual(M.semanticView(two,4).nodes.map(n=>n.id),['shared','two']);
assert.equal(two.links.length,1);
const limited=M.selectView(g,{mode:'overview',showDetails:true,importanceLimit:1});
assert.deepEqual(M.semanticView(limited,4,'one').nodes.map(n=>n.id),['shared','two']);
""")

    def test_multiple_documents_and_components_do_not_create_taxonomy_or_cross_links(self):
        self.js("""
const v=view([concept('r1',1,'learning',{sources:[{document_id:'first'}]}),
 concept('d1',1,'detail',{sources:[{document_id:'first'}]}),
 concept('r2',1,'learning',{sources:[{document_id:'second'}]}),
 concept('d2',2,'detail',{sources:[{document_id:'second'}]})],
 [{id:'first',source:'d1',target:'r1',relation:'PART_OF'},
  {id:'second',source:'r2',target:'d2',relation:'DRIVES'}]);
assert.deepEqual(M.tierProfile(v).cumulative,[2,2,3,4]);
assert.deepEqual(M.semanticView(v,1).links,[]);
assert.deepEqual(M.semanticView(v,3).links.map(e=>e.id),['first']);
assert.deepEqual(M.semanticView(v,4).links.map(e=>e.id),['first','second']);
assert.equal(M.semanticView(v,4).links[1].relation,'DRIVES');
""")

    def test_equal_importance_does_not_invent_ranks_to_fill_four_tiers(self):
        self.js("""
const same=view(Array.from({length:12},(_,i)=>concept('n'+i,2)),[{source:'n0',target:'n1',relation:'RELATED_TO'}]);
assert.deepEqual(M.tierProfile(same).counts,[0,12,0,0]);
assert.equal(M.tierProfile(same).minTier,2);
assert.equal(M.semanticView(same,1).nodes.length,0);
assert.equal(M.semanticView(same,2).nodes.length,12);
const allMain=view(Array.from({length:12},(_,i)=>concept('n'+i,1)));
assert.deepEqual(M.tierProfile(allMain).counts,[12,0,0,0]);
const allDetail=view([concept('detail',1,'detail'),concept('fine',1,'detail')]);
assert.deepEqual(M.tierProfile(allDetail).counts,[0,0,2,0]);
assert.equal(M.tierProfile(allDetail).minTier,3);
assert.equal(M.semanticView(allDetail,1).nodes.length,0);
assert.equal(M.semanticView(allDetail,3).nodes.length,2);
""")

    def test_special_ids_rejected_nodes_and_empty_graph(self):
        self.js("""
const g=M.normalize({nodes:[concept(0),concept('__proto__',1,'detail'),concept('constructor',2),
 concept('rejected',1,'learning',{review_status:'rejected'})],links:[
 {source:0,target:'__proto__',relation:'HAS_PART'},
 {source:'rejected',target:'constructor',relation:'RELATED_TO'}]});
const profile=M.tierProfile(g);
assert.equal(Object.getPrototypeOf(profile.byId),null);
assert.equal(profile.byId['0'],1);
assert.equal(profile.byId['__proto__'],3);
assert.equal(profile.byId.constructor,2);
assert.equal(profile.total,3);
assert.deepEqual(M.semanticView(g,1,'__proto__').nodes.map(n=>n.id),['0','__proto__']);
assert.deepEqual(M.semanticView(g,1,'rejected').nodes.map(n=>n.id),['0']);
const empty=M.normalize({concepts:[],relationships:[]});
assert.deepEqual(M.tierProfile(empty).cumulative,[0,0,0,0]);
assert.equal(M.tierProfile(empty).minTier,null);
assert.deepEqual(M.semanticView(empty,4).nodes,[]);
assert.deepEqual(M.semanticView(empty,4).links,[]);
""")

    def test_large_multidocument_graph_has_nested_sets_and_keeps_distinct_same_name_ids(self):
        self.js("""
const nodes=[],links=[];
for(let document=0;document<24;document++){
  for(let index=0;index<60;index++){
    const id='doc'+document+':concept'+index;
    const detail=index%4>=2,importance=index%2===0?1:2;
    nodes.push(concept(id,importance,detail?'detail':'learning',{
      name:'Shared display name '+(index%5),sources:[{document_id:'doc'+document,page_no:index+1}]}));
    if(index)links.push({id:'edge'+document+':'+index,source:'doc'+document+':concept'+(index-1),target:id,
      relation:index%2?'PART_OF':'CONNECTS_TO',review_status:index%3?'pending':'approved'});
  }
}
// These are actual cross-document associations, not inferred hierarchy.
for(let document=1;document<24;document++)links.push({id:'cross'+document,
 source:'doc'+(document-1)+':concept0',target:'doc'+document+':concept0',relation:'RELATED_TO'});
const v=view(nodes,links),before=JSON.stringify(v),profile=M.tierProfile(v);
assert.equal(profile.total,1440);
assert.deepEqual(profile.counts,[360,360,360,360]);
assert.equal(new Set(v.nodes.map(n=>n.name)).size,5);
assert.equal(Object.keys(profile.byId).length,1440);
let previous=new Set();const originals=new Map(v.links.map(edge=>[edge.id,edge]));
for(let tier=1;tier<=4;tier++){
  const visible=M.semanticView(v,tier),ids=new Set(visible.nodes.map(n=>n.id));
  for(const id of previous)assert.ok(ids.has(id));
  assert.equal(visible.nodes.length,360*tier);
  assert.equal(visible.links.length,profile.linkCounts[tier-1]);
  for(const edge of visible.links){
    assert.ok(ids.has(edge.source)&&ids.has(edge.target));
    assert.equal(edge,originals.get(edge.id));
  }
  previous=ids;
}
assert.equal(M.semanticView(v,4).links.length,links.length);
const filtered=view(nodes,links,{documentId:'doc7'});
assert.equal(M.tierProfile(filtered).total,60);
assert.ok(M.semanticView(filtered,4,'doc8:concept0').nodes.every(n=>n.id.startsWith('doc7:')));
assert.equal(M.semanticView(filtered,4).links.length,59);
assert.equal(JSON.stringify(v),before);
""")

    def test_pure_functions_keep_nodes_edges_and_audit_metadata_unchanged(self):
        self.js("""
const raw={nodes:[concept('main'),concept('detail',2,'detail')],links:[
 {id:'edge',source:'detail',target:'main',relation:'PART_OF',review_status:'pending',description:'Evidence from notes'}],
 review_items:[{type:'review',reason:'Teacher needed'}],hierarchy_candidates:[]};
const graph=M.normalize(raw),v=M.selectView(graph,{mode:'overview',showDetails:true});
const beforeRaw=JSON.stringify(raw),beforeGraph=JSON.stringify(graph),beforeView=JSON.stringify(v);
freeze(raw);freeze(graph);freeze(v);
M.tierProfile(graph);M.tierProfile(v);const s=M.semanticView(v,1,'detail');
assert.equal(JSON.stringify(raw),beforeRaw);
assert.equal(JSON.stringify(graph),beforeGraph);
assert.equal(JSON.stringify(v),beforeView);
assert.equal(s.nodes[1],v.nodes[1]);
assert.equal(s.links[0],v.links[0]);
assert.equal(s.links[0].reviewStatus,'pending');
assert.equal(s.nodes[1].importanceLevel,2);
assert.deepEqual(graph.reviewItems,[{type:'review',reason:'Teacher needed'}]);
""")


if __name__ == "__main__":
    unittest.main()
