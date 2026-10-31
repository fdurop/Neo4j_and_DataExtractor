/* Bounded local spring simulation; no graph records or relationships are changed. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.CourseGraphSprings=api;})(typeof window!=='undefined'?window:this,function(){
 'use strict';
 function create(nodes,edges){
  const particles=new Map(nodes.map(n=>[n.id,{...n,homeX:n.x,homeY:n.y,vx:0,vy:0}]));
  const springs=edges.filter(e=>particles.has(e.source)&&particles.has(e.target)).map(e=>{const a=particles.get(e.source),b=particles.get(e.target);return {...e,rest:Math.max(70,Math.hypot(a.x-b.x,a.y-b.y))};});
  return {particles,springs};
 }
 function step(model,anchors){
  for(const [id,position] of anchors || []){const p=model.particles.get(id);if(p){p.x=position.x;p.y=position.y;}}
  const forces=new Map([...model.particles].map(([id,p])=>[id,{x:(p.homeX-p.x)*.003,y:(p.homeY-p.y)*.003}]));
  for(const edge of model.springs){const a=model.particles.get(edge.source),b=model.particles.get(edge.target),dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),f=(d-edge.rest)*.025;
   forces.get(a.id).x+=f*dx/d;forces.get(a.id).y+=f*dy/d;forces.get(b.id).x-=f*dx/d;forces.get(b.id).y-=f*dy/d;
  }
  const nodes=[...model.particles.values()];
  for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){const a=nodes[i],b=nodes[j],dx=b.x-a.x||.01,dy=b.y-a.y||.01,d=Math.max(1,Math.hypot(dx,dy));if(d<75){const f=(75-d)*.06;forces.get(a.id).x-=f*dx/d;forces.get(a.id).y-=f*dy/d;forces.get(b.id).x+=f*dx/d;forces.get(b.id).y+=f*dy/d;}}
  let energy=0;
  for(const [id,p] of model.particles){if(p.fixed)continue;const force=forces.get(id);p.vx=(p.vx+Math.max(-5,Math.min(5,force.x)))*.84;p.vy=(p.vy+Math.max(-5,Math.min(5,force.y)))*.84;
   const limit=Math.max(1,Math.hypot(p.vx,p.vy)/14);p.vx/=limit;p.vy/=limit;p.x+=p.vx;p.y+=p.vy;energy+=p.vx*p.vx+p.vy*p.vy;
  }
  return energy;
 }
 return {create,step};
});
