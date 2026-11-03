"""Search keeps exactly one-hop context without writes or network access."""
import unittest
import test_graph_semantic_zoom as semantic_checks

class GraphSearchTests(unittest.TestCase):
    js = semantic_checks.GraphSemanticZoomTests.js
    def test_incoming_outgoing_details_and_no_second_hop(self):
        self.js("""
const v=view([concept('标准CAN标识符',2,'detail'),concept('a',3),concept('peer'),concept('second'),concept('unrelated')],
 [{source:'a',target:'标准CAN标识符',relation:'PREREQUISITE_OF'},
  {source:'标准CAN标识符',target:'peer',relation:'HAS_PART'},
  {source:'peer',target:'second',relation:'RELATED_TO'},
  {source:'a',target:'peer',relation:'RELATED_TO'}],{search:'标准CAN标识符',showDetails:false});
assert.deepEqual(v.nodes.map(n=>n.id),['标准CAN标识符','a','peer']);
assert.equal(v.links.length,2);
assert.deepEqual(M.semanticView(v,1).nodes.map(n=>n.id),['标准CAN标识符','a','peer']);
assert.deepEqual(view([concept('a')],[],{search:'missing'}).nodes,[]);
""")

    def test_relation_status_and_mode_stay_authoritative(self):
        self.js("""
const nodes=[concept('match'),concept('pending'),concept('rejected'),concept('approved'),concept('material')];
const links=[{source:'match',target:'pending',relation:'RELATED_TO'},
 {source:'match',target:'rejected',relation:'RELATED_TO',review_status:'rejected'},
 {source:'approved',target:'match',relation:'PREREQUISITE_OF',review_status:'approved'},
 {source:'match',target:'material',relation:'HAS_EVIDENCE'}];
assert.deepEqual(view(nodes,links,{search:'match',showPending:false}).nodes.map(n=>n.id),['match','approved']);
assert.deepEqual(view(nodes,links,{search:'match',mode:'prerequisites'}).nodes.map(n=>n.id),['match','approved']);
assert.deepEqual(view(nodes,links,{search:'match',importanceLimit:1}).nodes.map(n=>n.id),['match','pending','approved']);
""")

    def test_search_modes_keep_context_metadata_without_hiding_full_graph(self):
        self.js("""
const nodes=[concept('hit'),concept('neighbor',3,'detail'),concept('other')];
const links=[{source:'neighbor',target:'hit',relation:'RELATED_TO'},{source:'neighbor',target:'other',relation:'REQUIRES'}];
for(const searchMode of ['highlight','fade']){
 const v=view(nodes,links,{search:'hit',searchMode});
 assert.equal(v.nodes.length,3);assert.equal(v.links.length,2);
 assert.deepEqual(v.searchMatchIds,['hit']);assert.deepEqual(v.searchContextIds,['hit','neighbor']);
 assert.equal(v.searchMode,searchMode);assert.equal(M.semanticView(v,1).nodes.length,3);
 const cleared=view(nodes,links,{search:'',searchMode});assert.equal(cleared.searchActive,false);assert.deepEqual(cleared.searchMatchIds,[]);
}
const v=view(nodes,links,{search:'hit'});assert.equal(v.nodes.length,2);assert.equal(v.links.length,1);
""")
