const assert = require('node:assert/strict');
const physics = require('../frontend/static/js/graph_springs.js');
const model = physics.create([{id:'a',x:0,y:0,fixed:true},{id:'b',x:200,y:0,fixed:false},{id:'pinned',x:700,y:100,fixed:true}], [{source:'a',target:'b'}]);
const speeds=[];
for(let i=0;i<240;i++){
  physics.step(model,[['a',{x:80,y:0}],['pinned',{x:700,y:100}]]);
  speeds.push(model.particles.get('b').vx);
}
const b=model.particles.get('b');
assert(b.x>240 && b.x<290,'Neighbor follows the dragged anchor');
assert(speeds.some(v=>v>0) && speeds.some(v=>v<0),'Motion overshoots then springs back');
assert(Math.abs(b.vx)<.01,'Damping eventually settles the graph');
assert.deepEqual([model.particles.get('pinned').x,model.particles.get('pinned').y],[700,100]);
assert.equal(model.particles.get('a').x,80,'Released anchor stays where the user dropped it');
console.log('Spring following, overshoot, damping and fixed anchors passed');
