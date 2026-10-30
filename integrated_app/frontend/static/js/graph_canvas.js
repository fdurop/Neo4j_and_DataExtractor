/* Graph viewport and vector export. Cytoscape.js is vendored separately under MIT. */
(function (root, factory) {
  const value = factory();
  if (typeof module === "object" && module.exports) module.exports = value;
  else root.CourseGraphCanvas = value;
})(typeof window !== "undefined" ? window : this, function () {
  "use strict";
  const COLORS = {ink:"#27343b",muted:"#7b858c",line:"#97a6af",accent:"#266b60",paper:"#ffffff"};
  function escape(value) {return String(value == null ? "" : value).replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&apos;"})[c]);}
  function fitText(value, length) {const chars=Array.from(String(value));return chars.length>length ? chars.slice(0,length-1).join("")+"…" : chars.join("");}
  function fitLabel(value, width, fontSize) {
    const chars=Array.from(String(value)),budget=Math.max(1,(width-28)/(fontSize || 14));let used=0,result="";
    for(const char of chars){const size=/[\u0000-\u007f]/.test(char) ? .62 : 1;if(used+size>budget-1)return result+"…";used+=size;result+=char;}
    return result;
  }
  function palette(role) {
    return role.confirmed && role.key === "learning" ? {fill:"#edf5f1",border:"#81a696"} :
      role.key === "learning" ? {fill:"#ffffff",border:"#a6b8b0"} : {fill:"#f5f6f7",border:"#d0d6da"};
  }
  function scene(view, layout, roleInfo) {
    const nodes = view.nodes.map(node=>{
      const size=layout.nodeSizes && layout.nodeSizes[node.id] || {width:layout.nodeWidth,height:layout.nodeHeight,fontSize:14};
      return {id:node.id,name:node.name,description:node.description,displayTier:node.displayTier,
        role:roleInfo(node),width:size.width,height:size.height,fontSize:size.fontSize,
        position:{x:layout.positions[node.id].x+size.width/2,y:layout.positions[node.id].y+size.height/2}};
    });
    const names=new Map(nodes.map(n=>[n.id,n.name]));
    const edges=view.links.map(edge=>({id:"edge:"+edge.id,recordId:edge.id,source:edge.displaySource || edge.source,
      target:edge.displayTarget || edge.target,label:view.mode==="hierarchy" ? edge.relation==="IS_A" ? "分类" : "包含" :
        ({PREREQUISITE_OF:"先修",HAS_PART:"包含",PART_OF:"组成部分",IS_A:"是一种",CONTROLS:"控制",DRIVES:"驱动",HAS_PARAMETER:"参数",FLOWS_TO:"流向",REQUIRES:"运行需要",IMPLEMENTS:"实现",CONNECTS_TO:"连接",RELATED_TO:"相关",APPLIES_TO:"应用于"})[edge.relation] || edge.relation,
      description:edge.description,approved:edge.reviewStatus==="approved"}));
    return {nodes,edges,names,width:layout.nodeWidth || 240,height:layout.nodeHeight || 76,groups:layout.groups || [],mode:view.mode};
  }
  function bounds(nodes, width, height, groups) {
    if(!nodes.length)return {x:0,y:0,width:640,height:420};
    const lefts=nodes.map(n=>n.position.x-(n.width || width)/2),tops=nodes.map(n=>n.position.y-(n.height || height)/2),rights=nodes.map(n=>n.position.x+(n.width || width)/2),bottoms=nodes.map(n=>n.position.y+(n.height || height)/2);
    let left=Math.min(...lefts),top=Math.min(...tops),right=Math.max(...rights),bottom=Math.max(...bottoms);
    for(const group of groups || []){left=Math.min(left,group.x);top=Math.min(top,group.y);right=Math.max(right,group.x+group.width);bottom=Math.max(bottom,group.y+group.height);}
    return {x:left,y:top,width:right-left,height:bottom-top};
  }
  function edgeGeometry(source,target,width,height,targetWidth,targetHeight) {
    const dx=target.x-source.x,dy=target.y-source.y;
    const ratio=Math.min(Math.abs(dx)>0 ? width/2/Math.abs(dx) : Infinity,Math.abs(dy)>0 ? height/2/Math.abs(dy) : Infinity);
    const targetRatio=Math.min(Math.abs(dx)>0 ? (targetWidth || width)/2/Math.abs(dx) : Infinity,Math.abs(dy)>0 ? (targetHeight || height)/2/Math.abs(dy) : Infinity);
    const endRatio=Math.max(0,1-targetRatio);
    const start={x:source.x+dx*ratio,y:source.y+dy*ratio},end={x:source.x+dx*endRatio,y:source.y+dy*endRatio};
    const bend=(end.x-start.x)*.5;
    return {path:`M ${start.x} ${start.y} C ${start.x+bend} ${start.y}, ${end.x-bend} ${end.y}, ${end.x} ${end.y}`,
      label:{x:(start.x+end.x)/2,y:(start.y+end.y)/2-9}};
  }
  function svgMarkup(model, options) {
    options=options || {};
    const b=bounds(model.nodes,model.width,model.height,model.groups),pad=36,header=options.title ? 62 : 0;
    const width=Math.ceil(b.width+pad*2),height=Math.ceil(b.height+pad*2+header),offsetX=pad-b.x,offsetY=pad+header-b.y;
    const positions=new Map(model.nodes.map(n=>[n.id,n]));
    const output=[`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="SimSun, 宋体, serif">`,
      `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
      `<defs><marker id="kg-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 Z" fill="${COLORS.line}"/></marker></defs>`];
    if(options.title)output.push(`<text x="36" y="31" font-size="18" font-weight="bold" fill="${COLORS.ink}">${escape(options.title)}</text>`,
      `<text x="36" y="54" font-size="11" fill="${COLORS.muted}">${escape(options.subtitle || "")}</text>`);
    output.push(`<g transform="translate(${offsetX} ${offsetY})">`);
    for(const group of model.groups || [])output.push(`<rect x="${group.x}" y="${group.y}" width="${group.width}" height="${group.height}" rx="4" fill="#fcfcfc" stroke="#e1e5e8"/>`,
      `<text x="${group.x+16}" y="${group.y+24}" font-size="12" fill="${COLORS.muted}">${escape(group.label)}</text>`);
    for(const edge of model.edges){const source=positions.get(edge.source),target=positions.get(edge.target);if(!source || !target)continue;
      const line=edgeGeometry(source.position,target.position,source.width || model.width,source.height || model.height,target.width || model.width,target.height || model.height);
      output.push(`<path d="${line.path}" fill="none" stroke="${COLORS.line}" stroke-width="1.25"${edge.approved ? "" : ' stroke-dasharray="5 4"'} marker-end="url(#kg-arrow)"/>`);
      if(options.showEdgeLabels)output.push(`<text x="${line.label.x}" y="${line.label.y}" text-anchor="middle" font-size="11" fill="${COLORS.muted}">${escape(edge.label)}</text>`);
    }
    for(const node of model.nodes){const p=palette(node.role),w=node.width || model.width,h=node.height || model.height,fs=node.fontSize || 14,x=node.position.x-w/2,y=node.position.y-h/2;
      output.push(`<g><title>${escape(node.name+" · "+node.role.label)}</title><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${p.fill}" stroke="${p.border}"/>`,
        `<rect x="${x}" y="${y+12}" width="3" height="${h-24}" rx="1.5" fill="${node.role.key==="learning" ? COLORS.accent : "#c4ccd1"}"/>`,
        `<text x="${node.position.x}" y="${node.position.y+5}" text-anchor="middle" font-size="${fs}" fill="${COLORS.ink}">${escape(fitLabel(node.name,w,fs))}</text></g>`);
    }
    output.push("</g></svg>");
    return {markup:output.join(""),width,height};
  }
  function styles() {return [
        {selector:"node",style:{shape:"round-rectangle",width:"data(width)",height:"data(height)","background-color":"data(fill)",
          "border-color":"data(border)","border-width":1,label:"data(label)",color:COLORS.ink,"font-size":14,"font-weight":500,
          "font-family":'"Segoe UI", "Microsoft YaHei", sans-serif',"text-valign":"center","text-halign":"center", "text-wrap":"ellipsis",
          "text-max-width":"data(textWidth)","min-zoomed-font-size":5,"text-background-opacity":0,"overlay-opacity":0}},
        {selector:"node.group",style:{shape:"round-rectangle",label:"data(label)","font-size":12,"text-valign":"top","text-halign":"left","text-margin-y":-10,
          "background-color":"#fcfcfc","border-color":"#e1e5e8","border-width":1,padding:26,"text-wrap":"none","min-zoomed-font-size":5,events:"no"}},
        {selector:'node[tier = 1]',style:{"font-size":18,"font-weight":600,"border-width":1.8}},
        {selector:'node[tier = 2]',style:{"font-size":16}},
        {selector:'node[tier = 3]',style:{"font-size":14}},
        {selector:'node[tier = 4]',style:{"font-size":12}},
        {selector:"node:selected",style:{"border-color":COLORS.accent,"border-width":2.2,"background-color":"#edf5f1"}},
        {selector:"edge",style:{width:1.5,"line-color":COLORS.line,"target-arrow-color":COLORS.line,"target-arrow-shape":"triangle",
          "arrow-scale":.75,"curve-style":"bezier","line-style":"data(lineStyle)",label:"data(shownLabel)","font-size":11,color:COLORS.muted,
          "text-rotation":"none","text-margin-y":-9,"text-background-opacity":0,"text-outline-width":0,"min-zoomed-font-size":8,"overlay-opacity":0}},
        {selector:"edge.active, edge.hovered",style:{"line-color":COLORS.accent,"target-arrow-color":COLORS.accent,width:1.8,label:"data(relationLabel)",color:COLORS.accent}}
      ];}
  function viewportSubset(model, spatial, tier, rectangle, selectedId) {
    const allowed=node=>!node.displayTier || node.displayTier<=tier || node.id===selectedId;
    const lookup=model.nodeById || new Map(model.nodes.map(node=>[node.id,node]));
    const ids=new Set(spatial.query(rectangle).filter(id=>lookup.has(id) && allowed(lookup.get(id))));
    if(selectedId && lookup.has(selectedId))ids.add(selectedId);
    const nodes=Array.from(ids,id=>lookup.get(id)),edges=[],seen=new Set();
    const candidates=model.incident ? nodes.flatMap(node=>model.incident.get(node.id) || []) : model.edges;
    for(const edge of candidates)if(!seen.has(edge.id) && ids.has(edge.source) && ids.has(edge.target)){seen.add(edge.id);edges.push(edge);}
    return {nodes,edges};
  }
  function create(container, supplied) {
    const options=supplied || {}, engine=options.cytoscape, logic=options.model, map=options.mapLayout;
    if(typeof engine!=="function" || !logic || !map)throw Error("图谱绘制组件未加载");
    const cy=engine({container,elements:[],layout:{name:"preset"},minZoom:.015,maxZoom:5,
      boxSelectionEnabled:false,autoungrabify:true,style:styles()});
    let model=null,sourceView=null,profile=null,spatial=null,baseZoom=1,currentTier=1,selected=null,
      showAll=false,automatic=false,showEdgeLabels=false,sceneKey="",guard=false,queued=false,notificationKey="";
    const tooltip=document.createElement("div");tooltip.className="kg-canvas-tooltip";tooltip.hidden=true;container.append(tooltip);
    function updateZoom(){if(options.onZoom)options.onZoom(cy.zoom()/baseZoom);}
    function nodeElement(node){const color=palette(node.role);return {data:{id:node.id,label:node.name,fill:color.fill,border:color.border,tier:node.displayTier || 3,
      width:node.width,height:node.height,textWidth:node.width-24,hint:node.name+"\n"+node.role.label+(node.description ? "\n"+fitText(node.description,80) : "")},position:node.position,classes:"concept"};}
    function edgeElement(edge){return {data:{id:edge.id,source:edge.source,target:edge.target,relationLabel:edge.label,
      shownLabel:showEdgeLabels ? edge.label : "",lineStyle:edge.approved ? "solid" : "dashed",
      hint:model.names.get(edge.source)+" → "+model.names.get(edge.target)+"\n"+edge.label+" · "+(edge.approved ? "教师已确认" : "待确认")+(edge.description ? "\n"+edge.description : "")}};}
    function paintSelection(){cy.batch(()=>{cy.elements().unselect().removeClass("active");const node=cy.getElementById(selected || "");if(node.length){node.select();node.connectedEdges().addClass("active");}});}
    function refresh(){
      queued=false;if(!model || guard)return;
      currentTier=automatic && !showAll ? Math.max(profile.minTier || 1,logic.zoomTier(cy.zoom()/baseZoom,currentTier)) : 4;
      const extent=cy.extent(),padding=Math.min(900,100/Math.max(cy.zoom(),.01))+240;
      const subset=viewportSubset(model,spatial,currentTier,{x:extent.x1-padding,y:extent.y1-padding,width:extent.w+padding*2,height:extent.h+padding*2},selected);
      const nodeIds=new Set(subset.nodes.map(n=>n.id)),edgeIds=new Set(subset.edges.map(e=>e.id));
      cy.batch(()=>{
        cy.edges().filter(edge=>!edgeIds.has(edge.id())).remove();cy.nodes().filter(node=>!nodeIds.has(node.id())).remove();
        for(const node of subset.nodes)if(!cy.getElementById(node.id).length)cy.add(nodeElement(node));
        for(const edge of subset.edges)if(!cy.getElementById(edge.id).length)cy.add(edgeElement(edge));
      });paintSelection();
      const key=currentTier+"|"+subset.nodes.map(n=>n.id).join("|")+"|"+subset.edges.length;
      if(key!==notificationKey && options.onVisible){notificationKey=key;
        const tierView=logic.semanticView(sourceView,currentTier,selected);
        options.onVisible({tier:currentTier,automatic:automatic&&!showAll,nodes:subset.nodes,links:subset.edges,
          tierNodeCount:tierView.nodes.length,tierLinkCount:tierView.links.length,totalNodes:model.nodes.length,totalLinks:model.edges.length});
      }
    }
    function schedule(){tooltip.hidden=true;updateZoom();if(!queued && !guard){queued=true;requestAnimationFrame(refresh);}}
    cy.on("zoom pan",schedule);
    cy.on("tap","node.concept",event=>{if(options.onSelect)options.onSelect(event.target.id());});
    cy.on("tap","edge",event=>{if(options.onSelect)options.onSelect(event.target.source().id());});
    cy.on("mouseover","node.concept,edge",event=>{
      const target=event.target,p=event.renderedPosition || target.renderedPosition();if(!p)return;
      tooltip.textContent=target.data("hint");tooltip.style.left=Math.max(8,Math.min(p.x+16,container.clientWidth-240))+"px";
      tooltip.style.top=Math.max(8,Math.min(p.y+16,container.clientHeight-90))+"px";tooltip.hidden=false;
      if(target.isEdge())target.addClass("hovered");
    });
    cy.on("mouseout","node.concept,edge",event=>{tooltip.hidden=true;event.target.removeClass("hovered");});
    const resize=typeof ResizeObserver!=="undefined" ? new ResizeObserver(()=>{cy.resize();schedule();}) : null;
    if(resize)resize.observe(container);
    function home(fitAll){if(!model)return;
      if(!model.nodes.length){
        baseZoom=1;currentTier=1;guard=true;cy.zoom(1);cy.pan({x:0,y:0});guard=false;
        notificationKey="";refresh();updateZoom();return;
      }
      const main=automatic ? model.nodes.filter(n=>n.displayTier<=(profile.minTier || 1)) : model.nodes;
      const b=bounds(main.length ? main : model.nodes,model.width,model.height);
      const fitZoom=Math.max(.015,Math.min(1,(cy.width()-100)/Math.max(1,b.width),(cy.height()-100)/Math.max(1,b.height)));
      // A whole course must not make the initial labels microscopic. A separate
      // overview action can still show the entire map; its detail tier stays low.
      baseZoom=Math.max(.6,fitZoom);
      const zoom=fitAll ? fitZoom : baseZoom,center={x:b.x+b.width/2,y:b.y+b.height/2};
      if(zoom>fitZoom && main.length){
        const nearest=main.reduce((a,n)=>Math.hypot(n.position.x-center.x,n.position.y-center.y)<Math.hypot(a.position.x-center.x,a.position.y-center.y) ? n : a);
        center.x=nearest.position.x;center.y=nearest.position.y;
      }
      currentTier=profile.minTier || 1;guard=true;cy.zoom(zoom);cy.pan({x:cy.width()/2-center.x*zoom,y:cy.height()/2-center.y*zoom});guard=false;
      notificationKey="";refresh();updateZoom();
    }
    return {
      render(view,layout,settings){
        const config=settings || {},key=view.mode+"|"+view.nodes.map(n=>n.id).join("|")+"|"+view.links.map(e=>e.id).join("|");
        const retain=key===sceneKey;sceneKey=key;sourceView=view;profile=logic.tierProfile(view);
        const originals=new Map(view.nodes.map(node=>[node.id,node]));model=scene(view,layout,config.roleInfo);model.nodes.forEach(node=>{node.displayTier=logic.displayTier(originals.get(node.id));});
        model.nodeById=new Map(model.nodes.map(node=>[node.id,node]));model.incident=new Map(model.nodes.map(node=>[node.id,[]]));
        for(const edge of model.edges){model.incident.get(edge.source)?.push(edge);model.incident.get(edge.target)?.push(edge);}
        spatial=map.spatialIndex(model.nodes.map(n=>({id:n.id,x:n.position.x-n.width/2,y:n.position.y-n.height/2,width:n.width,height:n.height})),320);
        automatic=!!config.semanticZoom;showAll=!!config.showAll;showEdgeLabels=!!config.showEdgeLabels;selected=config.selected;
        cy.elements().remove();cy.resize();notificationKey="";if(retain){refresh();updateZoom();}else home();
      },
      select(id,focus){selected=id;if(model)refresh();paintSelection();if(focus){const node=cy.getElementById(id || "");if(node.length)cy.center(node);}},
      setEdgeLabels(value){showEdgeLabels=!!value;cy.edges().forEach(edge=>edge.data("shownLabel",value ? edge.data("relationLabel") : ""));},
      zoomBy(factor){cy.zoom({level:Math.min(cy.maxZoom(),Math.max(cy.minZoom(),cy.zoom()*factor)),renderedPosition:{x:cy.width()/2,y:cy.height()/2}});},
      resetZoom(){home(false);},
      fit(){home(true);},
      exportSVG(settings){if(!model)throw Error("没有可导出的图谱");const ids=new Set(logic.semanticView(sourceView,currentTier,selected).nodes.map(n=>n.id));
        return svgMarkup(Object.assign({},model,{nodes:model.nodes.filter(n=>ids.has(n.id)),edges:model.edges.filter(e=>ids.has(e.source)&&ids.has(e.target))}),Object.assign({showEdgeLabels},settings));},
      destroy(){if(resize)resize.disconnect();cy.destroy();}
    };
  }
  return {create,scene,bounds,edgeGeometry,svgMarkup,styles,viewportSubset};
});
