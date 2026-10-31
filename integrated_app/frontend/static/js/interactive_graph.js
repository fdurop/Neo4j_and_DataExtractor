/* Interactive force-directed editor, built on the vendored Cytoscape engine. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory;else root.CourseGraphCanvas=factory(root.CourseGraphCanvas);})(typeof window!=='undefined'?window:this,function(exports){
 'use strict';
 function radialPositions(center,ids){const result=new Map();let offset=0,ring=0;while(offset<ids.length){const count=Math.min(12+ring*8,ids.length-offset),radius=180+ring*160;for(let i=0;i<count;i++){const angle=-Math.PI/2+2*Math.PI*i/count+(ring%2?Math.PI/count:0);result.set(ids[offset+i],{x:center.x+radius*Math.cos(angle),y:center.y+radius*Math.sin(angle)});}offset+=count;ring++;}return result;}

 function create(container,options){
  const engine=options.cytoscape,storageKey='fdurop:positions:v1:'+(new URLSearchParams(location.search).get('course_id')||'import');
  let view=null,selected=null,layout=null,key='',paused=false,settings={},positions={};
  try{positions=JSON.parse(localStorage.getItem(storageKey)||'{}');}catch(_){}
  const cy=engine({container,elements:[],layout:{name:'preset'},minZoom:.08,maxZoom:4,autoungrabify:false,boxSelectionEnabled:false,style:[
   {selector:'node',style:{shape:'ellipse',width:58,height:58,'background-color':'#e6f0ed','border-color':'#65998d','border-width':2,label:'data(label)',color:'#233e37','font-family':'Microsoft YaHei, sans-serif','font-size':13,'text-valign':'bottom','text-margin-y':9,'text-wrap':'wrap','text-max-width':110,'overlay-opacity':0}},
   {selector:'node:selected',style:{'background-color':'#b9ded0','border-color':'#155e50','border-width':4}},
   {selector:'node.dimmed',style:{opacity:.15,'text-opacity':0}},
   {selector:'edge.dimmed',style:{opacity:.06,'text-opacity':0}},
   {selector:'node:locked',style:{'border-style':'double','border-width':5}},
   {selector:'edge',style:{width:1.5,'line-color':'#9babba','target-arrow-color':'#9babba','target-arrow-shape':'triangle','curve-style':'bezier','line-style':'data(lineStyle)',label:'data(label)','font-size':10,color:'#52616d','text-background-color':'#fff','text-background-opacity':.9,'text-background-padding':3,'text-rotation':'autorotate','overlay-padding':10}},
   {selector:'edge:selected, edge.active',style:{width:3,'line-color':'#176957','target-arrow-color':'#176957',label:'data(relationLabel)'}}
  ]});
  let radialSnapshot=null,radialCamera=null;
  function remember(){cy.nodes().forEach(n=>{const original=radialSnapshot?.get(n.id());positions[n.id()]=original?{...original}:{...n.position(),fixed:n.locked()};});try{localStorage.setItem(storageKey,JSON.stringify(positions));}catch(_){} }
  function visible(){if(!view)return;options.onVisible?.({tier:4,automatic:false,nodes:view.nodes,links:view.links,tierNodeCount:view.nodes.length,tierLinkCount:view.links.length,totalNodes:view.nodes.length,totalLinks:view.links.length});}
  function arrange(randomize){layout?.stop();if(!cy.nodes().length)return;const reduce=typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
   layout=cy.layout({name:'cose',randomize:!!randomize,animate:!reduce,animationDuration:450,refresh:20,numIter:400,nodeRepulsion:()=>12000,idealEdgeLength:()=>150,edgeElasticity:()=>90,gravity:.12,componentSpacing:100,nodeDimensionsIncludeLabels:true,fit:true,padding:60});
   layout.one('layoutstop',()=>{remember();visible();});layout.run();
  }
  function moveNode(node,position,animate,fixed){node.stop(true,false);if(node.locked())node.unlock();if(animate&&!fixed&&!reduced()&&!cy.headless())node.animate({position},{duration:240,easing:'ease-in-out-cubic'});else node.position(position);if(fixed)node.lock();}
  function restoreRadial(){if(!radialSnapshot)return;const previous=radialSnapshot,camera=radialCamera;radialSnapshot=null;radialCamera=null;cy.batch(()=>{for(const [id,position] of previous){const n=cy.getElementById(id);if(n.length)moveNode(n,{x:position.x,y:position.y},false,position.fixed);}});if(camera){cy.zoom(camera.zoom);cy.pan(camera.pan);}}
  function spreadRadial(node){if(!radialSnapshot){radialSnapshot=new Map(cy.nodes().map(n=>[n.id(),{...n.position(),fixed:n.locked()}]));radialCamera={zoom:cy.zoom(),pan:{...cy.pan()}};}
   const neighbors=node.connectedEdges().connectedNodes().not(node).sort((a,b)=>a.id().localeCompare(b.id()));
   const targets=radialPositions(node.position(),neighbors.map(n=>n.id()));
   cy.batch(()=>{for(const [id,position] of targets)moveNode(cy.getElementById(id),position,true,radialSnapshot.get(id).fixed);});
   if(!cy.headless()&&neighbors.length){let width=cy.width(),height=cy.height();const panel=typeof document!=='undefined'?document.getElementById('inspectorDrawer'):null;if(panel&&panel.getClientRects().length){const box=panel.getBoundingClientRect();if(width>850)width-=box.width;else height=Math.max(160,height-box.height);}
    const radius=Math.max(...[...targets.values()].map(p=>Math.hypot(p.x-node.position().x,p.y-node.position().y)))+80;
    const zoom=Math.max(cy.minZoom(),Math.min(radialCamera.zoom,Math.max(160,width-80)/(radius*2),Math.max(160,height-80)/(radius*2)));
    cy.zoom(zoom);cy.pan({x:Math.max(160,width)/2-node.position().x*zoom,y:Math.max(160,height)/2-node.position().y*zoom});
   }
  }
  function choose(id){id=id||null;if(id!==selected||(!radialSnapshot&&id)){stopSprings(false);layout?.stop();restoreRadial();selected=id;const node=cy.getElementById(id||'');if(node.length)spreadRadial(node);}
   cy.batch(()=>{cy.elements().unselect().removeClass('active dimmed');const n=cy.getElementById(id||'');if(n.length){const edges=n.connectedEdges(),neighbors=edges.connectedNodes().union(n);n.select();edges.addClass('active');cy.nodes().not(neighbors).addClass('dimmed');cy.edges().not(edges).addClass('dimmed');}});}

  cy.on('tap','node',e=>{options.onSelect?.(e.target.id());});
  cy.on('tap',e=>{if(e.target===cy)options.onSelect?.(null);});
  cy.on('tap','edge',e=>{cy.elements().unselect();e.target.select();options.onEdgeSelect?.(e.target.data('recordId'));});
  let springModel=null,frame=null,holding=false,deadline=0;
  const clock=()=>typeof performance!=='undefined'?performance.now():Date.now();
  const reduced=()=>typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
  function stopSprings(save){if(frame!==null){cancelAnimationFrame(frame);frame=null;}springModel=null;holding=false;if(save)remember();}
  function tick(){frame=null;if(!springModel||paused||reduced()){stopSprings(true);return;}
   const anchors=[];for(const [id,p] of springModel.particles)if(p.fixed){const n=cy.getElementById(id);if(n.length)anchors.push([id,n.position()]);}
   const energy=window.CourseGraphSprings.step(springModel,anchors);
   cy.batch(()=>{for(const [id,p] of springModel.particles)if(!p.fixed){const n=cy.getElementById(id);if(n.length&&!n.locked()&&!n.grabbed())n.position({x:p.x,y:p.y});}});
   if(holding||(clock()<deadline&&energy>.002))frame=requestAnimationFrame(tick);else stopSprings(true);
  }
  cy.on('grab','node',event=>{layout?.stop();stopSprings(false);if(paused||reduced()||typeof window==='undefined'||!window.CourseGraphSprings)return;
   const dragged=event.target,near=dragged.union(dragged.neighborhood('node')),affected=near.union(near.neighborhood('node')).slice(0,60),ids=new Set(affected.map(n=>n.id()));
   springModel=window.CourseGraphSprings.create(affected.map(n=>({id:n.id(),...n.position(),fixed:n.id()===dragged.id()||n.locked()})),cy.edges().filter(e=>ids.has(e.source().id())&&ids.has(e.target().id())).map(e=>({source:e.source().id(),target:e.target().id()})));
   holding=true;frame=requestAnimationFrame(tick);
  });
  cy.on('dragfree','node',()=>{holding=false;if(radialSnapshot&&selected){stopSprings(false);const node=cy.getElementById(selected);if(node.length)spreadRadial(node);remember();}else{deadline=clock()+2000;if(!springModel)remember();}});
  cy.on('cxttap','node',e=>{e.target.stop(true,true);e.target.locked()?e.target.unlock():e.target.lock();if(radialSnapshot?.has(e.target.id()))radialSnapshot.get(e.target.id()).fixed=e.target.locked();remember();});
  cy.on('zoom',()=>options.onZoom?.(cy.zoom()));
  const resize=typeof ResizeObserver!=='undefined'?new ResizeObserver(()=>cy.resize()):null;resize?.observe(container);
  return {
   render(next,unused,config){restoreRadial();remember();settings=config||{};view=next;const nextKey=next.mode+'|'+next.nodes.map(n=>n.id).join('|')+'|'+next.links.map(e=>e.id).join('|'),changed=key!==nextKey;key=nextKey;
    stopSprings(false);layout?.stop();const z=cy.zoom(),pan={...cy.pan()};cy.elements().remove();
    cy.add(next.nodes.map((n,i)=>{const p=positions[n.id]||{x:Math.cos(i*2.399)*Math.sqrt(i+1)*65,y:Math.sin(i*2.399)*Math.sqrt(i+1)*65};return {data:{id:n.id,label:n.name},position:{x:p.x,y:p.y},locked:!!p.fixed};}));
    const ids=new Set(next.nodes.map(n=>n.id));cy.add(next.links.filter(e=>ids.has(e.displaySource||e.source)&&ids.has(e.displayTarget||e.target)).map(e=>({data:{id:'edge:'+e.id,recordId:e.id,source:e.displaySource||e.source,target:e.displayTarget||e.target,relationLabel:e.relation,label:settings.showEdgeLabels?e.relation:'',lineStyle:e.reviewStatus==='approved'?'solid':'dashed'}})));
    cy.resize();visible();if(changed&&next.nodes.some(n=>!positions[n.id]))arrange(false);else if(changed)cy.fit(cy.elements(),60);else{cy.zoom(z);cy.pan(pan);}choose(config.selected); },
   select(id,focus){choose(id);if(focus&&!radialSnapshot&&cy.getElementById(id).length)cy.animate({center:{eles:cy.getElementById(id)},duration:200});},
   setEdgeLabels(value){settings.showEdgeLabels=value;cy.edges().forEach(e=>e.data('label',value?e.data('relationLabel'):''));},
   zoomBy(factor){cy.zoom({level:Math.max(.08,Math.min(4,cy.zoom()*factor)),renderedPosition:{x:cy.width()/2,y:cy.height()/2}});},
   resetZoom(){cy.zoom(1);const n=cy.getElementById(selected||"");if(n.length)cy.center(n);else if(cy.nodes().length)cy.center(cy.nodes().sort((a,b)=>b.degree()-a.degree())[0]);},fit(){cy.fit(cy.elements(),60);},
   relayout(){if(selected&&options.onSelect){options.onSelect(null);if(selected)return;}restoreRadial();stopSprings(false);positions={};cy.nodes().unlock();arrange(true);},
   pause(){paused=!paused;if(paused){layout?.stop();stopSprings(true);}return paused;},
   pin(){stopSprings(true);const n=cy.getElementById(selected||'');if(!n.length)return null;n.stop(true,true);n.locked()?n.unlock():n.lock();if(radialSnapshot?.has(n.id()))radialSnapshot.get(n.id()).fixed=n.locked();remember();return n.locked();},
   exportSVG(config){if(!view)throw Error('没有可导出的图谱');const nodes=view.nodes.map(n=>({id:n.id,name:n.name,position:cy.getElementById(n.id).position(),width:80,height:60,fontSize:13,role:settings.roleInfo(n)}));return exports.svgMarkup({nodes,edges:view.links.map(e=>({source:e.displaySource||e.source,target:e.displayTarget||e.target,label:e.relation,approved:e.reviewStatus==='approved'})),width:80,height:60,groups:[]},Object.assign({showEdgeLabels:settings.showEdgeLabels},config));},
   destroy(){stopSprings(false);restoreRadial();layout?.stop();resize?.disconnect();cy.destroy();}
  };
 }
 return Object.assign({},exports,{create,radialPositions});
});
