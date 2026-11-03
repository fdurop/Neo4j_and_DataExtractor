/* Interactive force-directed editor, built on the vendored Cytoscape engine. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory;else root.CourseGraphCanvas=factory(root.CourseGraphCanvas);})(typeof window!=='undefined'?window:this,function(exports){
 'use strict';
 function radialPositions(center,ids,scale=1){const result=new Map();let offset=0,ring=0;while(offset<ids.length){const count=Math.min(12+ring*8,ids.length-offset),radius=(180+ring*160)*scale;for(let i=0;i<count;i++){const angle=-Math.PI/2+2*Math.PI*i/count+(ring%2?Math.PI/count:0);result.set(ids[offset+i],{x:center.x+radius*Math.cos(angle),y:center.y+radius*Math.sin(angle)});}offset+=count;ring++;}return result;}

 // Pack disconnected components as loose islands around the largest component.
 // Only presentation coordinates change; no synthetic edges are introduced.
 function organicComponents(groups){
  const ordered=groups.filter(g=>g.nodes.length).map(group=>{
   const xs=group.nodes.map(n=>n.x),ys=group.nodes.map(n=>n.y);
   const x=(Math.min(...xs)+Math.max(...xs))/2,y=(Math.min(...ys)+Math.max(...ys))/2;
   const radius=Math.max(...group.nodes.map(n=>Math.hypot(n.x-x,n.y-y)+(n.radius||65)));
   return {...group,x,y,radius};
  }).sort((a,b)=>Number(!!b.fixed)-Number(!!a.fixed)||b.nodes.length-a.nodes.length||b.radius-a.radius);
  const positions=new Map(),occupied=[],gap=42,step=86;
  for(const group of ordered){
   let x=group.x,y=group.y;
   if(!group.fixed){
    if(!occupied.length){x=0;y=0;}
    else {
     let attempt=0;
     do {const angle=attempt*2.399963229728653;
      const radius=step*Math.sqrt(attempt++);
      x=radius*Math.cos(angle);y=radius*Math.sin(angle);
     }while(occupied.some(other=>Math.hypot(x-other.x,y-other.y)<group.radius+other.radius+gap));
    }
   }
   occupied.push({x,y,radius:group.radius});
   for(const n of group.nodes)positions.set(n.id,{x:n.x+x-group.x,y:n.y+y-group.y});
  }
  return positions;
 }

 function create(container,options){
  const engine=options.cytoscape,storageKey='fdurop:positions:v1:'+(new URLSearchParams(location.search).get('course_id')||'import');
  let view=null,selected=null,layout=null,key='',paused=false,settings={},positions={};
  const layoutVersion='organic-20261101',versionKey=storageKey+':layout';
  let needsLayout=true;try{needsLayout=localStorage.getItem(versionKey)!==layoutVersion;}catch(_){}
  try{positions=JSON.parse(localStorage.getItem(storageKey)||'{}');}catch(_){}
  const cy=engine({container,elements:[],layout:{name:'preset'},minZoom:.08,maxZoom:4,autoungrabify:false,boxSelectionEnabled:false,style:[
   {selector:'node',style:{shape:'ellipse',width:'data(diameter)',height:'data(diameter)','background-color':'data(fill)','border-color':'data(border)','border-width':2,label:n=>nodeLabel(n),color:'#233e37','font-family':'Microsoft YaHei, sans-serif','font-size':n=>nodeFont(n),'font-weight':n=>n.data('importance')===1?'bold':'normal','text-outline-color':'#ffffff','text-outline-width':2,'text-valign':'bottom','text-margin-y':9,'text-wrap':'wrap','text-max-width':110,'overlay-opacity':0}},
   {selector:'node:selected',style:{'border-color':'#183b65','border-width':4}},
   {selector:'node.dimmed',style:{opacity:.15,'text-opacity':0}},
   {selector:'edge.dimmed',style:{opacity:.06,'text-opacity':0}},
   {selector:'node:locked',style:{'border-style':'double','border-width':5}},
   {selector:'edge',style:{width:1.5,'line-color':'#9babba','target-arrow-color':'#9babba','target-arrow-shape':'triangle','curve-style':'bezier','line-style':'data(lineStyle)',label:'data(label)','font-size':10,color:'#52616d','text-background-color':'#fff','text-background-opacity':.9,'text-background-padding':3,'text-rotation':'autorotate','overlay-padding':10}},
   {selector:'node.search-match',style:{'border-color':'#d97706','border-width':6,'background-color':'#ffd58a',width:'data(searchDiameter)',height:'data(searchDiameter)'}},
   {selector:'node.search-muted',style:{opacity:.16,'text-opacity':.2}},
   {selector:'edge.search-muted',style:{opacity:.07,'text-opacity':0}},
   {selector:'edge:selected, edge.active',style:{width:3,'line-color':'#176957','target-arrow-color':'#176957',label:'data(relationLabel)'}}
  ]});
  let radialSnapshot=null;
  const important=n=>n.hasClass('search-match')||n.data('priority')||n.selected()||n.hasClass('focus-peer')||n.hasClass('hovered');
  function nodeLabel(n){return important(n)||cy.zoom()>=.5||(cy.zoom()>=.25&&n.data('importance')<=2)?n.data('label'):'';}
  function nodeFont(n){return important(n)?Math.max(n.data('fontSize'),Math.min(140,11/cy.zoom())):n.data('fontSize');}
  let legend=null,tooltip=null;
  if(container?.parentElement&&typeof document!=='undefined'){
   legend=document.createElement('div');legend.className='graph-visual-legend';
   Object.assign(legend.style,{position:'absolute',left:'14px',top:'14px',zIndex:4,padding:'8px 11px',borderRadius:'6px',background:'rgba(255,255,255,.94)',color:'#50616f',fontSize:'11px',pointerEvents:'none',boxShadow:'0 1px 5px #233e3712'});
   legend.innerHTML='<span style="color:#376fa4">● 核心</span>　<span style="color:#4d8f87">● 次要</span>　<span style="color:#a0afbc">● 细节</span><span style="margin-left:12px">重要度为估计 · 大小表示关联节点数</span>';
   container.parentElement.appendChild(legend);
   const filterLegend=document.querySelector('.filter-panel .legend');if(filterLegend)filterLegend.innerHTML=legend.innerHTML+'<span style="margin-left:12px">┄ 待确认关系</span>';
   tooltip=document.createElement('div');tooltip.setAttribute('role','tooltip');
   Object.assign(tooltip.style,{position:'absolute',zIndex:8,padding:'9px 12px',borderRadius:'6px',background:'#243b4a',color:'white',fontSize:'12px',whiteSpace:'pre-line',pointerEvents:'none',maxWidth:'260px',display:'none'});
   container.parentElement.appendChild(tooltip);
  }
  function refreshLabels(){cy.nodes().updateStyle();}
  cy.on('mouseover','node',e=>{const n=e.target;n.addClass('hovered');refreshLabels();if(tooltip){tooltip.textContent=n.data('label')+'\n重要度估计：'+['','核心','次要','细节'][n.data('importance')]+' · 关联 '+n.data('neighborCount')+' 个节点';tooltip.style.display='block';}});
  cy.on('mousemove','node',e=>{if(tooltip){const p=e.renderedPosition||e.target.renderedPosition();tooltip.style.left=Math.max(4,Math.min(cy.width()-280,p.x+18))+'px';tooltip.style.top=Math.max(4,Math.min(cy.height()-85,p.y+18))+'px';}});
  cy.on('mouseout','node',e=>{e.target.removeClass('hovered');if(tooltip)tooltip.style.display='none';refreshLabels();});
  function remember(){cy.nodes().forEach(n=>{const original=radialSnapshot?.get(n.id());positions[n.id()]=original?{...original}:{...n.position(),fixed:n.locked()};});try{localStorage.setItem(storageKey,JSON.stringify(positions));}catch(_){} }
  function visible(){if(!view)return;options.onVisible?.({tier:4,automatic:false,nodes:view.nodes,links:view.links,tierNodeCount:view.nodes.length,tierLinkCount:view.links.length,totalNodes:view.nodes.length,totalLinks:view.links.length});}
  let layoutReady=Promise.resolve();
  function arrange(randomize){layout?.stop();if(!cy.nodes().length)return;const reduce=typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
   const pinned=new Map(cy.nodes(':locked').map(n=>[n.id(),{...n.position()}]));
   layout=cy.layout({name:'cose',randomize:!!randomize,animate:!reduce,animationDuration:450,refresh:20,numIter:400,nodeRepulsion:()=>24000,idealEdgeLength:()=>195,edgeElasticity:()=>90,gravity:.075,componentSpacing:160,nodeDimensionsIncludeLabels:true,fit:true,padding:60});
   let resolveLayout;layoutReady=new Promise(resolve=>{resolveLayout=resolve;});
   layout.one('layoutstop',()=>{
    cy.batch(()=>{
     for(const [id,point] of pinned){const n=cy.getElementById(id);n.unlock();n.position(point);n.lock();}
     const targets=organicComponents(cy.elements().components().map(component=>({
      fixed:component.nodes(':locked').length>0,
      nodes:component.nodes().map(n=>{const box=n.boundingBox({includeLabels:true});return {id:n.id(),...n.position(),radius:Math.max(65,Math.hypot(box.w,box.h)/2)};})
     })));
     for(const [id,point] of targets){const n=cy.getElementById(id);if(!n.locked())n.position(point);}
    });
    needsLayout=false;try{localStorage.setItem(versionKey,layoutVersion);}catch(_){}
    cy.fit(cy.elements(),65);remember();visible();resolveLayout();
   });layout.run();
  }
  function moveNode(node,position,animate,fixed){node.stop(true,false);if(node.locked())node.unlock();if(animate&&!fixed&&!reduced()&&!cy.headless())node.animate({position},{duration:240,easing:'ease-in-out-cubic'});else node.position(position);if(fixed)node.lock();}
  function restoreRadial(){if(!radialSnapshot)return;const previous=radialSnapshot;radialSnapshot=null;cy.batch(()=>{for(const [id,position] of previous){const n=cy.getElementById(id);if(n.length)moveNode(n,{x:position.x,y:position.y},false,position.fixed);}});}
  function spreadRadial(node){if(!radialSnapshot)radialSnapshot=new Map(cy.nodes().map(n=>[n.id(),{...n.position(),fixed:n.locked()}]));
   const neighbors=node.connectedEdges().connectedNodes().not(node).sort((a,b)=>a.id().localeCompare(b.id()));
   const scale=Math.min(4,Math.max(1,.65/cy.zoom()));
   const targets=radialPositions(node.position(),neighbors.map(n=>n.id()),scale);
   cy.batch(()=>{for(const [id,position] of targets)moveNode(cy.getElementById(id),position,true,radialSnapshot.get(id).fixed);});
  }
  function paintSearch(){
   cy.elements().removeClass('search-match search-muted');
   if(!view?.searchActive)return;
   const matches=new Set(view.searchMatchIds || []),context=new Set(view.searchContextIds || []);
   cy.nodes().forEach(n=>{if(matches.has(n.id()))n.addClass('search-match');else if(view.searchMode==='fade'&&!context.has(n.id()))n.addClass('search-muted');});
   if(view.searchMode==='fade')cy.edges().forEach(e=>{if(!context.has(e.source().id())||!context.has(e.target().id())||(!matches.has(e.source().id())&&!matches.has(e.target().id())))e.addClass('search-muted');});
  }
  function choose(id){id=id||null;if(id!==selected||(!radialSnapshot&&id)){
   const clicked=cy.getElementById(id||''),anchor=clicked.length?{...clicked.position()}:null,baseline=radialSnapshot,camera={zoom:cy.zoom(),pan:{...cy.pan()}};
   stopSprings(false);layout?.stop();cy.zoom(camera.zoom);cy.pan(camera.pan);restoreRadial();selected=id;
   const node=cy.getElementById(id||'');if(node.length){if(baseline)radialSnapshot=baseline;if(anchor)moveNode(node,anchor,false,node.locked());spreadRadial(node);}
  }
   cy.batch(()=>{cy.elements().unselect().removeClass('active dimmed focus-peer');const n=cy.getElementById(id||'');if(n.length){const edges=n.connectedEdges(),neighbors=edges.connectedNodes().union(n);n.select();neighbors.addClass('focus-peer');edges.addClass('active');cy.nodes().not(neighbors).addClass('dimmed');cy.edges().not(edges).addClass('dimmed');}});paintSearch();refreshLabels();}

  cy.on('tap','node',e=>{
   const target=e.target,id=target.id(),focus=cy.getElementById(selected||'');
   const adjacent=focus.length&&focus.neighborhood('node').some(n=>n.id()===id);
   options.onSelect?.(selected&&(id===selected||!adjacent)?null:id);
  });
  cy.on('tap',e=>{if(e.target===cy)options.onSelect?.(null);});
  cy.on('tap','edge',e=>{
   const edge=e.target;
   if(selected&&edge.source().id()!==selected&&edge.target().id()!==selected){options.onSelect?.(null);return;}
   cy.edges().unselect();edge.select();options.onEdgeSelect?.(edge.data('recordId'));
  });
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
  cy.on('zoom',()=>{refreshLabels();options.onZoom?.(cy.zoom());});
  const resize=typeof ResizeObserver!=='undefined'?new ResizeObserver(()=>cy.resize()):null;resize?.observe(container);
  return {
   render(next,unused,config){restoreRadial();remember();settings=config||{};view=next;const nextKey=next.mode+'|'+next.nodes.map(n=>n.id).join('|')+'|'+next.links.map(e=>e.id).join('|'),changed=key!==nextKey;key=nextKey;
    stopSprings(false);layout?.stop();const z=cy.zoom(),pan={...cy.pan()};cy.elements().remove();
    const adjacency=new Map(next.nodes.map(n=>[n.id,new Set()]));
    for(const e of next.links){const a=e.displaySource||e.source,b=e.displayTarget||e.target;if(a!==b&&adjacency.has(a)&&adjacency.has(b)){adjacency.get(a).add(b);adjacency.get(b).add(a);}}
    const maxNeighbors=Math.max(1,...[...adjacency.values()].map(n=>n.size));
    const ranked=next.nodes.slice().sort((a,b)=>(a.importanceLevel||3)-(b.importanceLevel||3)||adjacency.get(b.id).size-adjacency.get(a.id).size);
    const priority=new Set(ranked.slice(0,Math.min(24,Math.max(6,Math.ceil(Math.sqrt(next.nodes.length))))).map(n=>n.id));
    const palette={1:['#376fa4','#245681'],2:['#77b6af','#4d8f87'],3:['#dce3e9','#a0afbc']};
    cy.add(next.nodes.map((n,i)=>{const p=positions[n.id]||{x:Math.cos(i*2.399)*Math.sqrt(i+1)*65,y:Math.sin(i*2.399)*Math.sqrt(i+1)*65};const importance=[1,2,3].includes(n.importanceLevel)?n.importanceLevel:3,count=adjacency.get(n.id).size;return {data:{id:n.id,label:n.name,importance,neighborCount:count,diameter:44+52*Math.log1p(count)/Math.log1p(maxNeighbors),searchDiameter:(44+52*Math.log1p(count)/Math.log1p(maxNeighbors))*1.22,fill:palette[importance][0],border:palette[importance][1],fontSize:importance===1?16:13,priority:priority.has(n.id)},position:{x:p.x,y:p.y},locked:!!p.fixed};}));
    const ids=new Set(next.nodes.map(n=>n.id));cy.add(next.links.filter(e=>ids.has(e.displaySource||e.source)&&ids.has(e.displayTarget||e.target)).map(e=>({data:{id:'edge:'+e.id,recordId:e.id,source:e.displaySource||e.source,target:e.displayTarget||e.target,relationLabel:e.relation,label:settings.showEdgeLabels?e.relation:'',lineStyle:e.reviewStatus==='approved'?'solid':'dashed'}})));
    cy.resize();visible();if(changed&&(needsLayout||next.nodes.some(n=>!positions[n.id])))arrange(false);else if(changed)cy.fit(cy.elements(),60);else{cy.zoom(z);cy.pan(pan);}choose(config.selected); },
   whenReady(){return layoutReady;},
   select(id){choose(id);},
   setEdgeLabels(value){settings.showEdgeLabels=value;cy.edges().forEach(e=>e.data('label',value?e.data('relationLabel'):''));},
   zoomBy(factor){cy.zoom({level:Math.max(.08,Math.min(4,cy.zoom()*factor)),renderedPosition:{x:cy.width()/2,y:cy.height()/2}});},
   resetZoom(){cy.zoom(1);const n=cy.getElementById(selected||"");if(n.length)cy.center(n);else if(cy.nodes().length)cy.center(cy.nodes().sort((a,b)=>b.degree()-a.degree())[0]);},fit(){cy.fit(cy.elements(),60);},
   relayout(){if(selected&&options.onSelect){options.onSelect(null);if(selected)return;}restoreRadial();stopSprings(false);positions={};cy.nodes().unlock();arrange(true);},
   pause(){paused=!paused;if(paused){layout?.stop();stopSprings(true);}return paused;},
   pin(){stopSprings(true);const n=cy.getElementById(selected||'');if(!n.length)return null;n.stop(true,true);n.locked()?n.unlock():n.lock();if(radialSnapshot?.has(n.id()))radialSnapshot.get(n.id()).fixed=n.locked();remember();return n.locked();},
   exportSVG(config){if(!view)throw Error('没有可导出的图谱');const nodes=view.nodes.map(n=>({id:n.id,name:n.name,position:cy.getElementById(n.id).position(),width:80,height:60,fontSize:13,role:settings.roleInfo(n)}));return exports.svgMarkup({nodes,edges:view.links.map(e=>({source:e.displaySource||e.source,target:e.displayTarget||e.target,label:e.relation,approved:e.reviewStatus==='approved'})),width:80,height:60,groups:[]},Object.assign({showEdgeLabels:settings.showEdgeLabels},config));},
   destroy(){legend?.remove();tooltip?.remove();stopSprings(false);restoreRadial();layout?.stop();resize?.disconnect();cy.destroy();}
  };
 }
 return Object.assign({},exports,{create,radialPositions,organicComponents});
});
