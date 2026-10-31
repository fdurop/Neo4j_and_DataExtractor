const assert=require('node:assert/strict');
const path=require('node:path');
const root=path.join(__dirname,'../frontend/static/js/');
global.location={search:'?course_id=focus-test'};
global.localStorage={getItem:()=>null,setItem:()=>{}};
global.matchMedia=()=>({matches:true});
const cytoscape=require(root+'vendor/cytoscape.min.js');
let cy,focus=null,canvas;
canvas=require(root+'interactive_graph.js')(require(root+'graph_canvas.js')).create(null,{
 cytoscape:options=>(cy=cytoscape({...options,container:undefined,headless:true,styleEnabled:true})),
 onSelect:id=>{focus=id===focus?null:id;canvas.select(focus);}
});
canvas.render({mode:'overview',nodes:['a','b','c','d'].map(id=>({id,name:id})),links:[['a','b'],['a','c'],['b','c']].map(([source,target],i)=>({id:String(i),source,target,relation:'RELATED_TO'}))},null,{});
const before=cy.nodes().map(n=>({...n.position()}));
cy.getElementById('a').emit('tap');
assert(cy.getElementById('a').selected());
assert(!cy.getElementById('b').hasClass('dimmed'));
assert(!cy.getElementById('c').hasClass('dimmed'));
assert(cy.getElementById('d').hasClass('dimmed'));
assert(cy.getElementById('edge:2').hasClass('dimmed'),'Neighbor-to-neighbor edges must fade');
assert(!cy.getElementById('edge:0').hasClass('dimmed'));
assert.deepEqual(cy.nodes().map(n=>({...n.position()})),before,'Focus must preserve positions');
cy.getElementById('a').emit('grab');cy.getElementById('a').emit('dragfree');
assert(cy.getElementById('a').selected(),'Dragging must not toggle selection');
cy.getElementById('a').emit('tap');assert.equal(cy.elements('.dimmed').length,0);
cy.getElementById('b').emit('tap');assert(cy.getElementById('b').selected());
cy.emit('tap');assert.equal(cy.elements('.dimmed').length,0);assert.equal(cy.nodes(':selected').length,0);
canvas.destroy();
console.log('One-hop focus, fading, toggle, background reset and stable positions passed');
