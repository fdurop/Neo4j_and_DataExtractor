/* Interactive force-directed editor, built on the vendored Cytoscape engine. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory;else root.CourseGraphCanvas=factory(root.CourseGraphCanvas);})(typeof window!=='undefined'?window:this,function(exports){
 'use strict';
 function create(container,options){
  const engine=options.cytoscape,storageKey='fdurop:positions:v1:'+(new URLSearchParams(location.search).get('course_id')||'import');
  let view=null,selected=null,layout=null,key='',paused=false,settings={},positions={};
  try{positions=JSON.parse(localStorage.getItem(storageKey)||'{}');}catch(_){}
  const cy=engine({container,elements:[],layout:{name:'preset'},minZoom:.08,maxZoom:4,autoungrabify:false,boxSelectionEnabled:false,style:[
   {selector:'node',style:{shape:'ellipse',width:58,height:58,'background-color':'#e6f0ed','border-color':'#65998d','border-width':2,label:'data(label)',color:'#233e37','font-family':'Microsoft YaHei, sans-serif','font-size':13,'text-valign':'bottom','text-margin-y':9,'text-wrap':'wrap','text-max-width':110,'overlay-opacity':0}},
   {selector:'node:selected',style:{'background-color':'#b9ded0','border-color':'#155e50','border-width':4}},
   {selector:'node:locked',style:{'border-style':'double','border-width':5}},
   {selector:'edge',style:{width:1.5,'line-color':'#9babba','target-arrow-color':'#9babba','target-arrow-shape':'triangle','curve-style':'bezier','line-style':'data(lineStyle)',label:'data(label)','font-size':10,color:'#52616d','text-background-color':'#fff','text-background-opacity':.9,'text-background-padding':3,'text-rotation':'autorotate','overlay-padding':10}},
   {selector:'edge:selected, edge.active',style:{width:3,'line-color':'#176957','target-arrow-color':'#176957',label:'data(relationLabel)'}}
  ]});
  function remember(){cy.nodes().forEach(n=>{positions[n.id()]={...n.position(),fixed:n.locked()};});try{localStorage.setItem(storageKey,JSON.stringify(positions));}catch(_){} }
  function visible(){if(!view)return;options.onVisible?.({tier:4,automatic:false,nodes:view.nodes,links:view.links,tierNodeCount:view.nodes.length,tierLinkCount:view.links.length,totalNodes:view.nodes.length,totalLinks:view.links.length});}
  function arrange(randomize){layout?.stop();if(!cy.nodes().length)return;const reduce=typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
   layout=cy.layout({name:'cose',randomize:!!randomize,animate:!reduce,animationDuration:450,refresh:20,numIter:400,nodeRepulsion:()=>12000,idealEdgeLength:()=>150,edgeElasticity:()=>90,gravity:.12,componentSpacing:100,nodeDimensionsIncludeLabels:true,fit:true,padding:60});
   layout.one('layoutstop',()=>{remember();visible();});layout.run();
  }
  function choose(id){selected=id;cy.elements().unselect().removeClass('active');const n=cy.getElementById(id||'');if(n.length){n.select();n.connectedEdges().addClass('active');}}
  cy.on('tap','node',e=>{choose(e.target.id());options.onSelect?.(e.target.id());});
  cy.on('tap','edge',e=>{cy.elements().unselect();e.target.select();options.onEdgeSelect?.(e.target.data('recordId'));});
  cy.on('grab','node',()=>{layout?.stop();});
  cy.on('drag','node',e=>{if(paused)return;const n=e.target; n.neighborhood('node').filter(x=>!x.locked()&&!x.grabbed()).forEach(x=>{const p=x.position(),a=n.position(),d=Math.hypot(p.x-a.x,p.y-a.y)||1;if(d>230)x.position({x:p.x+(a.x-p.x)*.025,y:p.y+(a.y-p.y)*.025});});});
  cy.on('dragfree','node',remember);
  cy.on('cxttap','node',e=>{e.target.locked()?e.target.unlock():e.target.lock();remember();});
  cy.on('zoom',()=>options.onZoom?.(cy.zoom()));
  const resize=typeof ResizeObserver!=='undefined'?new ResizeObserver(()=>cy.resize()):null;resize?.observe(container);
  return {
   render(next,unused,config){remember();settings=config||{};view=next;const nextKey=next.mode+'|'+next.nodes.map(n=>n.id).join('|')+'|'+next.links.map(e=>e.id).join('|'),changed=key!==nextKey;key=nextKey;
    layout?.stop();const z=cy.zoom(),pan={...cy.pan()};cy.elements().remove();
    cy.add(next.nodes.map((n,i)=>{const p=positions[n.id]||{x:Math.cos(i*2.399)*Math.sqrt(i+1)*65,y:Math.sin(i*2.399)*Math.sqrt(i+1)*65};return {data:{id:n.id,label:n.name},position:p,locked:!!p.fixed};}));
    const ids=new Set(next.nodes.map(n=>n.id));cy.add(next.links.filter(e=>ids.has(e.displaySource||e.source)&&ids.has(e.displayTarget||e.target)).map(e=>({data:{id:'edge:'+e.id,recordId:e.id,source:e.displaySource||e.source,target:e.displayTarget||e.target,relationLabel:e.relation,label:settings.showEdgeLabels?e.relation:'',lineStyle:e.reviewStatus==='approved'?'solid':'dashed'}})));
    cy.resize();choose(config.selected);visible();if(changed&&next.nodes.some(n=>!positions[n.id]))arrange(false);else if(changed)cy.fit(cy.elements(),60);else{cy.zoom(z);cy.pan(pan);} },
   select(id,focus){choose(id);if(focus&&cy.getElementById(id).length)cy.animate({center:{eles:cy.getElementById(id)},duration:200});},
   setEdgeLabels(value){settings.showEdgeLabels=value;cy.edges().forEach(e=>e.data('label',value?e.data('relationLabel'):''));},
   zoomBy(factor){cy.zoom({level:Math.max(.08,Math.min(4,cy.zoom()*factor)),renderedPosition:{x:cy.width()/2,y:cy.height()/2}});},
   resetZoom(){cy.zoom(1);const n=cy.getElementById(selected||"");if(n.length)cy.center(n);else if(cy.nodes().length)cy.center(cy.nodes().sort((a,b)=>b.degree()-a.degree())[0]);},fit(){cy.fit(cy.elements(),60);},
   relayout(){positions={};cy.nodes().unlock();arrange(true);},
   pause(){paused=!paused;if(paused)layout?.stop();return paused;},
   pin(){const n=cy.getElementById(selected||'');if(!n.length)return null;n.locked()?n.unlock():n.lock();remember();return n.locked();},
   exportSVG(config){if(!view)throw Error('没有可导出的图谱');const nodes=view.nodes.map(n=>({id:n.id,name:n.name,position:cy.getElementById(n.id).position(),width:80,height:60,fontSize:13,role:settings.roleInfo(n)}));return exports.svgMarkup({nodes,edges:view.links.map(e=>({source:e.displaySource||e.source,target:e.displayTarget||e.target,label:e.relation,approved:e.reviewStatus==='approved'})),width:80,height:60,groups:[]},Object.assign({showEdgeLabels:settings.showEdgeLabels},config));},
   destroy(){layout?.stop();resize?.disconnect();cy.destroy();}
  };
 }
 return Object.assign({},exports,{create});
});
