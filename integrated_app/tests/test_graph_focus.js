const assert=require('node:assert/strict');
const path=require('node:path');
const root=path.join(__dirname,'../frontend/static/js/');
global.location={search:'?course_id=focus-test'};
let stored;
global.localStorage={getItem:()=>null,setItem:(key,value)=>{stored=JSON.parse(value);}};
global.matchMedia=()=>({matches:true});
const cytoscape=require(root+'vendor/cytoscape.min.js');
const api=require(root+'interactive_graph.js')(require(root+'graph_canvas.js'));
const rings=api.radialPositions({x:3,y:7},Array.from({length:35},(_,i)=>String(i)));
assert.equal(rings.size,35);
for(let i=0;i<35;i++){const p=rings.get(String(i));assert(Math.abs(Math.hypot(p.x-3,p.y-7)-(i<12?180:i<32?340:500))<1e-8);}
let cy,focus=null,canvas;
canvas=api.create(null,{
 cytoscape:options=>(cy=cytoscape({...options,container:undefined,headless:true,styleEnabled:true})),
 onSelect:id=>{focus=id===focus?null:id;canvas.select(focus);}
});
const graph={mode:'overview',nodes:['a','b','c','d'].map(id=>({id,name:id})),links:[['a','b'],['a','c'],['b','c']].map(([source,target],i)=>({id:String(i),source,target,relation:'RELATED_TO'}))};
canvas.render(graph,null,{});
const snapshot=()=>cy.nodes().map(n=>({...n.position()}));
const before=snapshot(),camera={zoom:cy.zoom(),pan:{...cy.pan()}};
const original=new Map(cy.nodes().map(n=>[n.id(),{...n.position()}]));
const sameOriginal=()=>assert.deepEqual(snapshot(),before,'Clearing focus must restore every original position');
cy.getElementById('a').emit('tap');
assert(cy.getElementById('a').selected());
assert(!cy.getElementById('b').hasClass('dimmed'));
assert(!cy.getElementById('c').hasClass('dimmed'));
assert(cy.getElementById('d').hasClass('dimmed'));
assert(cy.getElementById('edge:2').hasClass('dimmed'),'Neighbor-to-neighbor edges must fade');
assert(!cy.getElementById('edge:0').hasClass('dimmed'));
assert.deepEqual(cy.getElementById('a').position(),original.get('a'));
assert.deepEqual(cy.getElementById('d').position(),original.get('d'));
const center=cy.getElementById('a').position();
for(const id of ['b','c']){const p=cy.getElementById(id).position();assert(Math.abs(Math.hypot(p.x-center.x,p.y-center.y)-180)<1e-8);}
const b=cy.getElementById('b').position(),c=cy.getElementById('c').position();
assert(Math.abs(b.x+c.x-2*center.x)<1e-8&&Math.abs(b.y+c.y-2*center.y)<1e-8,'Two neighbors must be evenly opposed');
cy.getElementById('a').emit('grab');cy.getElementById('a').position({x:100,y:100});cy.getElementById('a').emit('dragfree');
assert(cy.getElementById('a').selected(),'Dragging must not toggle selection');
assert.deepEqual({x:stored.b.x,y:stored.b.y},original.get('b'),'Temporary radial positions must not be stored');
canvas.pin();assert(cy.getElementById('a').locked());
cy.getElementById('a').emit('tap');assert.equal(cy.elements('.dimmed').length,0);sameOriginal();
assert(cy.getElementById('a').locked(),'Explicit pin preferences survive restoration');
assert.deepEqual({zoom:cy.zoom(),pan:{...cy.pan()}},camera);
cy.getElementById('a').emit('tap');cy.getElementById('b').emit('tap');
assert.deepEqual(cy.getElementById('b').position(),original.get('b'),'Switching selection uses the original layout');
cy.emit('tap');sameOriginal();assert.equal(cy.nodes(':selected').length,0);
cy.getElementById('a').emit('tap');canvas.render(graph,null,{selected:'a'});cy.emit('tap');sameOriginal();
canvas.destroy();
console.log('Radial spacing, multi-ring layout, exact restoration, selection switching, dragging, storage, pinning, rerender and fading passed');
