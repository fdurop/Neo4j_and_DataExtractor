/* Course graph UI. The model keeps source organization, taxonomy and prerequisites separate. */
(function () {
  "use strict";
  const $ = id => document.getElementById(id), M = window.CourseGraphModel;
  const state = {raw:null,graph:null,view:null,mode:"overview",selected:null,local:false,dirty:false,edge:null,importLabel:"",loadPhase:"idle",loadError:"",loadErrorStatus:0};
  const labels = {PREREQUISITE_OF:"学习前置",HAS_PART:"包含",PART_OF:"组成部分",IS_A:"是一种",REQUIRES:"运行需要",RELATED_TO:"相关",APPLIES_TO:"应用于",HAS_PARAMETER:"具有参数",DRIVES:"驱动",CONTROLS:"控制",CONNECTS_TO:"连接",IMPLEMENTS:"实现",FLOWS_TO:"流向"};
  const array = value => Array.isArray(value) ? value : [];
  const properties = record => Object.assign({},record && record.properties || {},record || {});
  const nodeById = id => state.graph && state.graph.nodes.find(n => n.id === id);
  let chart = null, requestVersion = 0, connectFrom = null;
  const params = new URLSearchParams(location.search);
  const courseName = params.get("course_name") || "";
  const courseId = params.get("course_id") || "";
  function element(tag,text,cls) { const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);if(cls)e.className=cls;return e; }
  function message(text,error) { $("graphStatus").textContent=text;$("graphStatus").classList.toggle("error",!!error); }
  function button(text,action,cls) {const e=element("button",text,cls);e.type="button";e.addEventListener("click",action);return e;}
  function autosize(input){if(!input)return;input.style.height='0px';const height=Math.max(36,Math.min(240,input.scrollHeight+2));input.style.height=height+'px';input.style.overflowY=input.scrollHeight+2>240?'auto':'hidden';}
  function field(parent,title,tag,value,id) {const label=element("label",title,"field"),input=element(tag);if(id)input.id=id;input.value=value || "";label.append(input);parent.append(label);if(tag==="textarea"){input.rows=1;input.addEventListener("input",()=>autosize(input));requestAnimationFrame(()=>autosize(input));}return input;}
  function details(parent,title) {const e=element("details"),s=element("summary",title);e.append(s);parent.append(e);return e;}
  function selectOptions(select,items,value) {select.replaceChildren();for(const item of items){const o=element("option",item[1]);o.value=item[0];select.append(o);}select.value=value == null ? "" : String(value);}
  function documentLabel(id) {const d=state.graph.documents.find(d=>String(d.document_id || d.id)===String(id));return d ? d.file_name || d.chapter_title || d.name || id : id || "来源课件待补充";}
  function refreshDocuments() {const value=$("documentFilter").value;selectOptions($("documentFilter"),[["","全部课件"]].concat(state.graph.documents.map(d=>[String(d.document_id || d.id),d.file_name || d.chapter_title || d.name || d.document_id || d.id])),value);}
  async function api(path,method,payload) {
    const target = new URL(path, location.origin);
    if (courseId) target.searchParams.set("course_id", courseId);
    const response=await fetch(target.pathname+target.search,{method:method || "GET",credentials:"same-origin",headers:Object.assign(payload ? {"Content-Type":"application/json"} : {}, method && method!=="GET" && !state.local ? {"X-Graph-Revision":String(state.raw?.review_revision || 0)} : {}),body:payload ? JSON.stringify(payload) : undefined});
    let result;try {result=await response.json();}catch(_){throw Error("服务返回了无法识别的内容，请检查图谱接口。");}
    if(!response.ok || result.error || result.success===false){const error=Error(result.message || (result.error && result.error.message) || result.error || "图谱操作失败（"+response.status+"）");error.status=response.status;throw error;}
    return result;
  }
  function apply(raw,options) {
    const graph=M.normalize(raw);
    state.raw=raw;state.graph=graph;
    state.courseCounts={nodes:graph.nodes.filter(n=>n.kind==="concept").length,
      links:M.selectView(graph,{mode:"overview",showDetails:true,showPending:true,importanceLimit:3}).links.length};
    if(options && "local" in options)state.local=options.local;
    if(options && "dirty" in options)state.dirty=options.dirty;
    if(options && options.ready){state.loadPhase="ready";state.loadError="";state.loadErrorStatus=0;}
    if(state.selected && !nodeById(state.selected))state.selected=null;
    const title=state.raw.course && state.raw.course.course_name || state.raw.course_manifest && state.raw.course_manifest.course_name;
    const localTitle=title && title!=="default_course" ? title : graph.documents.length===1 ? (graph.documents[0].file_name || graph.documents[0].chapter_title) : null;
    $("courseName").textContent=state.local ? localTitle || state.importLabel || "导入的课程图谱" : courseName || title || "课程知识图谱";
    $("workspaceNote").textContent=state.local ? "正在编辑导入副本 · 修改保留在当前页面，请保存图谱 JSON。" : "正在查看课程数据 · 编辑后保存到当前课程。";
    $("workspaceNote").classList.toggle("local",state.local);
    refreshDocuments();render();renderInspector();
  }
  async function loadGraph() {
    if(state.dirty && !confirm("当前有未导出的修改。仍要重新加载课程图谱吗？"))return false;
    if(!courseName && !courseId){message("请从课程页面进入，或使用“导入图谱”打开已有JSON。");return false;}
    const version=++requestVersion;
    state.loadPhase="loading";state.loadError="";state.loadErrorStatus=0;renderEmpty();
    $("reloadGraph").disabled=true;message("正在加载课程图谱……");
    try {const raw=await api("/api/graph?course_name="+encodeURIComponent(courseName));if(version!==requestVersion)return false;state.selected=null;apply(raw,{local:false,dirty:false,ready:true});message("课程图谱已加载。");return true;}
    catch(error){if(version===requestVersion){state.loadPhase="error";state.loadError=error.message;state.loadErrorStatus=error.status || 0;renderEmpty();message(error.status===401 ? "请先登录，再重新进入课程图谱。" : "加载失败："+error.message+" 也可以导入已有图谱。",true);}return false;}
    finally {if(version===requestVersion)$("reloadGraph").disabled=false;}
  }
  async function reloadAfterMutation(text) {
    const previous=state.raw,version=requestVersion;
    const loaded=await loadGraph();
    if(!loaded && (state.raw!==previous || requestVersion!==version+1))return false;
    message(loaded ? text : "修改已提交，但重新加载未完成。请重新加载课程确认结果。",!loaded);
    return loaded;
  }
  async function mutateAndReload(path,method,payload,text) {
    const previous=state.raw,version=requestVersion;
    try {await api(path,method,payload);}catch(error){if(state.raw!==previous || requestVersion!==version)return false;throw error;}
    if(state.raw!==previous || requestVersion!==version || state.local)return false;
    return reloadAfterMutation(text);
  }
  function setInspector(open){document.querySelector(".workspace").classList.toggle("inspector-open",open);$("toggleInspector").setAttribute("aria-expanded",String(open));}
  function setFilters(open){$("filterPanel").hidden=!open;$("toggleFilters").setAttribute("aria-expanded",String(open));}
  function renderEmpty(){
    const view=state.view || {nodes:[],links:[]},presentation=M.emptyPresentation({phase:state.loadPhase,status:state.loadErrorStatus,error:state.loadError,
      hasNodes:!!view.nodes.length,conceptCount:state.courseCounts && state.courseCounts.nodes,mode:state.mode,courseSelected:!!(courseId || courseName)});
    $("graphContainer").setAttribute("aria-busy",String(state.loadPhase==="loading"));
    const previous=$("emptyAction");if(previous)previous.remove();
    $("graphEmpty").hidden=!presentation;
    if(!presentation)return;
    $("emptyTitle").textContent=presentation.title;$("emptyDescription").textContent=presentation.description;
    let action;
    if(presentation.action==="login" || presentation.action==="courses"){
      action=element("a",presentation.action==="login" ? "前往登录" : "返回课程列表","primary");action.href=presentation.action==="login" ? "/login/" : "/teacher/";
    }else if(presentation.action==="retry")action=button("重新加载课程图谱",loadGraph);
    else if(presentation.action==="clear")action=button("清空筛选",()=>{$("graphSearch").value="";$("documentFilter").value="";$("importanceFilter").value="3";render();});
    else if(presentation.action==="overview")action=button("查看课程概览",()=>{state.mode="overview";render();});
    if(presentation.kind==="hierarchy" && !$("showDetails").checked){
      const expanded=M.selectView(state.graph,{mode:"hierarchy",search:$("graphSearch").value,showDetails:true,showPending:$("showPending").checked,documentId:$("documentFilter").value,importanceLimit:Number($("importanceFilter").value)});
      if(expanded.links.length){$("emptyTitle").textContent="层级关系已被筛选收起";$("emptyDescription").textContent="这些关系连接了被标为素材细节的概念，可以展开后核对。";
        action=button("展开 "+expanded.links.length+" 条层级关系",()=>{$("showDetails").checked=true;render();$("toggleFilters").focus();});}
    }
    if(action){action.id="emptyAction";$("graphEmpty").append(action);}
  }
  function renderNodeList(nodes){
    const list=$("nodeList");list.replaceChildren();
    for(const n of nodes.slice(0,200)){const b=button(n.name,()=>choose(n.id,true,true));b.dataset.nodeId=n.id;b.setAttribute("aria-pressed",String(state.selected===n.id));list.append(b);}
    if(nodes.length>200)list.append(element("p","此档共 "+nodes.length+" 个节点，列表显示前200个；可搜索定位其他知识点。"));
    if(!nodes.length)list.append(element("p","当前视图没有可选择的概念。"));
  }
  function onVisible(info){
    if(!state.view)return;state.visibility=info;
    if(state.listView!==state.view || state.listTier!==info.tier || state.listSelected!==state.selected){
      renderNodeList(M.semanticView(state.view,info.tier,state.selected).nodes);
      state.listView=state.view;state.listTier=info.tier;state.listSelected=state.selected;
    }
    $("detailLevel").textContent=info.automatic ? ["主干","知识点","结构细节","全部细节"][info.tier-1]+" · "+info.tier+"/4" : "全部显示";
    $("graphSummary").textContent="视野 "+info.nodes.length+" 节点 / "+info.links.length+" 关系 · 本档 "+info.tierNodeCount+" / "+info.totalNodes+" 节点 · 整课 "+state.courseCounts.nodes+" 节点 / "+state.courseCounts.links+" 关系";
  }
  function choose(id,focus,toggle) {
    if(toggle && id===state.selected)id=null;
    if(connectFrom && id && id!==connectFrom){const source=connectFrom;connectFrom=null;showEdge(null,source);$("edgeTarget").value=id;return;}
    const previous=nodeById(state.selected),nameInput=$('editNodeName'),descriptionInput=$('editNodeDescription');
    if(toggle && id!==state.selected && previous && nameInput && descriptionInput && (nameInput.value!==previous.name || descriptionInput.value!==previous.description) && !confirm('当前节点有未保存的文字修改，仍要切换吗？')){if(chart)chart.select(state.selected,false);return;}
    state.selected=id;setInspector(!!id);if(chart)chart.select(id,!!focus);
    $("nodeList").querySelectorAll("[data-node-id]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.nodeId===id)));
    renderInspector();
  }
  function render() {
    if(!state.graph)return;
    const options={mode:state.mode,search:$("graphSearch").value,showDetails:state.mode==="overview" || $("showDetails").checked,showPending:$("showPending").checked,documentId:$("documentFilter").value,importanceLimit:Number($("importanceFilter").value)};
    const view=M.selectView(state.graph,options), layout=state.mode==="overview" ? window.CourseGraphMapLayout.layout(view,{tierOf:M.displayTier}) : M.layout(view,{mode:state.mode});
    state.view=view;
    document.querySelectorAll("[data-mode]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.mode===state.mode)));
    const descriptions={overview:"缩小查看主干，放大逐步显示知识点与细节。节点大小表示显示档位；所有连线来自已有关系，虚线待审核。四档为显示详略，非教师确认的教学深度。",hierarchy:"箭头：整体 / 上位概念 → 部分 / 下位概念。组成与分类分别标注，虚线关系仍待确认；可展开细节查看结构部件。",prerequisites:"箭头：先学 → 后学。补学顺序只使用教师确认的先修关系；运行依赖和课件顺序不作为学习先修。"};
    $("viewDescription").textContent=descriptions[state.mode];
    const total=state.graph.nodes.filter(n=>n.kind==="concept").length, confirmed=state.graph.nodes.filter(n=>M.roleInfo(n).confirmed && M.roleInfo(n).key==="learning").length;
    $("graphSummary").textContent="显示 "+view.nodes.length+" / "+total+" 个概念 · "+view.links.length+" 条当前关系 · 教师确认知识点 "+confirmed+" 个 · "+state.graph.hierarchyCandidates.length+" 条归组候选 · "+state.graph.reviewItems.length+" 条问题记录"+(state.dirty ? " · 有未导出的修改" : "");
    const warnings=Array.from(new Set((state.graph.warnings || []).concat(view.warnings || [],layout.warnings || [],chart ? [] : ["图形暂时不可用；仍可通过右侧列表查看、编辑知识点。请检查本机绘图组件是否完整。"])));
    $("graphWarning").hidden=!warnings.length;$("graphWarning").textContent=warnings.slice(0,5).join("\n")+(warnings.length>5 ? "\n另有"+(warnings.length-5)+"项数据提示，请检查导入文件。" : "");
    renderEmpty();
    renderNodeList(view.nodes);
    if(!chart){$("graphEmpty").hidden=false;$("emptyTitle").textContent="图形暂时不可用";$("emptyDescription").textContent="知识点列表与编辑仍可使用。请检查绘图组件，或重新加载页面。";return;}
    chart.render(view,layout,{roleInfo:M.roleInfo,selected:state.selected,showEdgeLabels:$("showEdgeLabels").checked,semanticZoom:state.mode==="overview",showAll:$("showDetails").checked});
  }

  function renderSources(parent,sources) {
    if(!sources.length){parent.append(element("p","尚未关联可核对的课件页。请补充带来源的建图结果。","muted"));return;}
    for(const source of sources){
      const box=element("div",undefined,"evidence");
      box.append(element("strong",documentLabel(source.document_id)+" · 第 "+(source.page_no == null ? "?" : source.page_no)+" 页"));
      const pdf=M.sourcePdfLink(state.raw,source);
      if(pdf){const paragraph=element("p"),link=element("a","打开课件第"+pdf.pageNo+"页");link.href=pdf.href;link.target="_blank";link.rel="noopener";paragraph.append(link);box.append(paragraph);}
      if(source.title)box.append(element("p",source.title,"muted"));
      if(source.extracted_text)details(box,"查看课件提取原文（待核对）").append(element("pre",source.extracted_text,"source-raw-text"));
      for(const e of array(source.material_evidence)){
        const whole=e.description_scope==="entire_page" || e.source_kind==="full_page";
        const local=e.description_scope==="embedded_image_only" || e.source_kind==="embedded_image";
        const label=e.node_type==="Table" ? "表格转录" : e.node_type==="Formula" ? "公式转录" : whole ? "整页视觉描述" : local ? "局部图片描述" : "关联素材描述";
        details(box,label+"（点击展开，需核对原图）").append(element("pre",e.text,"source-raw-text"));
        if(array(e.semantic_review_flags).length)box.append(element("p","这段素材包含待核对的方向、标签或对象归属描述，请结合原图确认。","notice warning"));
      }
      parent.append(box);
    }
  }
  function localChanged(name) {state.dirty=true;apply(state.raw,{dirty:true});if(name){const n=state.graph.nodes.find(n=>n.name===name);if(n){state.selected=n.id;render();renderInspector();}}message("修改已应用到导入副本，请保存图谱 JSON。");}
  function writeFields(record,patch) {
    const hadCanonical=Object.prototype.hasOwnProperty.call(record,"canonical_name"),nested=record.properties && typeof record.properties==="object";
    const nestedCanonical=nested && Object.prototype.hasOwnProperty.call(record.properties,"canonical_name");
    Object.assign(record,patch);if(nested)Object.assign(record.properties,patch);
    if(patch.name!==undefined){if(hadCanonical)record.canonical_name=patch.name;if(nestedCanonical)record.properties.canonical_name=patch.name;}
  }
  function renameLocalReferences(node,next) {
    const native=Array.isArray(state.raw.concepts);
    for(const edge of state.graph.links.concat(state.graph.hierarchyCandidates)) {
      for(const key of ["source","target"]) {
        if(edge[key]!==node.id)continue;
        for(const container of [edge.raw,edge.raw.properties]) {
          if(!container || !Object.prototype.hasOwnProperty.call(container,key))continue;
          const endpoint=container[key];
          if(endpoint && typeof endpoint==="object"){if(endpoint.name===node.name)endpoint.name=next;}
          else if(String(endpoint)===node.name && (native || node.name!==node.id))container[key]=next;
        }
      }
    }
    const previous=array(properties(node.raw).previous_names);
    writeFields(node.raw,{previous_names:Array.from(new Set(previous.concat(node.name))).filter(name=>name!==next)});
  }
  async function saveNode(node,patch,newName) {
    try {
      if(state.local) {
        if(patch.name!==undefined && patch.name!==node.name)renameLocalReferences(node,patch.name);
        writeFields(node.raw,patch);
        if(Array.isArray(state.raw.concepts) && patch.name!==undefined){node.raw.canonical_name=patch.name;}
        localChanged(newName || node.name);
      } else {
        const selectedName=newName || node.name;
        if(!await mutateAndReload("/api/graph/node","PUT",Object.assign({course_name:courseName,id:node.id},patch),"已保存到当前课程。"))return;
        const next=state.graph.nodes.find(n=>n.name===selectedName);if(next)choose(next.id);
      }
    } catch(error){message("保存失败："+error.message,true);}
  }
  function edgeRecord(edge) {const records=Array.isArray(state.raw.concepts) ? state.raw.relationships : state.raw.links;return {records,index:records.findIndex(r=>r===edge.raw || JSON.stringify(r)===JSON.stringify(edge.raw))};}
  function showEdge(edge,sourceId) {
    state.edge=edge || null;$("edgeTitle").textContent=edge ? "修改关系类型或依据" : "新增关系";
    const choices=state.graph.nodes.filter(n=>n.kind==="concept" && n.reviewStatus!=="rejected").map(n=>[n.id,n.name]);
    selectOptions($("edgeSource"),choices,edge ? edge.source : sourceId || choices[0] && choices[0][0]);
    selectOptions($("edgeTarget"),choices,edge ? edge.target : choices.find(x=>x[0]!==$("edgeSource").value)?.[0]);
    $("edgeSource").disabled=!!edge;$("edgeTarget").disabled=!!edge;
    if(edge && !Array.from($("edgeType").options).some(o=>o.value===edge.relation)){const o=element("option",edge.relation);o.value=edge.relation;$("edgeType").append(o);}
    $("edgeType").value=edge ? edge.relation : "PREREQUISITE_OF";$("edgeDescription").value=edge ? edge.description : "";$("edgeApproved").checked=!!edge && edge.reviewStatus==="approved";$("deleteSelectedEdge").hidden=!edge;$("edgeDialogError").textContent="";$("edgeDialog").showModal();
  }
  async function removeEdge(edge) {
    if(!confirm("删除这条关系？\n"+nodeById(edge.source).name+" → "+nodeById(edge.target).name))return;
    try {if(state.local){const found=edgeRecord(edge);if(found.index<0)throw Error("没有找到原始关系。");found.records.splice(found.index,1);localChanged();}
      else{await mutateAndReload("/api/graph/link","DELETE",{course_name:courseName,source_id:edge.source,target_id:edge.target,old_relation:edge.relation,relation_id:edge.raw.relation_id || edge.raw.id || edge.raw.relationship_id},"关系已删除。");}}
    catch(error){message("删除失败："+error.message,true);}
  }
  function renderInspector() {
    const parent=$("inspector"),node=nodeById(state.selected);parent.replaceChildren();
    if(!node){parent.append(element("h2","选择一个知识点"),element("p","点击图中节点或上方列表，查看教学用途、来源证据及前置知识。","muted"));return;}
    const role=M.roleInfo(node),raw=properties(node.raw),referenceNames=new Set([node.name].concat(array(raw.previous_names)));
    parent.append(element("h2",node.name),element("span",role.label,"badge "+(role.confirmed ? "confirmed" : role.key==="learning" ? "suggested" : "")));
    if(raw.model_update_pending)parent.append(element("p","模型有新的候选内容。当前教师决定已保留，请结合证据复核。","notice warning"));
    const name=field(parent,"名称","input",node.name,"editNodeName"),description=field(parent,"说明","textarea",node.description,"editNodeDescription");
    const save=button("保存文字",()=>{const next=name.value.trim();if(!next || state.graph.nodes.some(n=>n.id!==node.id && n.name===next)){message("名称不能为空或与当前课程其他节点重复。",true);return;}saveNode(node,{name:next,description:description.value.trim()},next);},"primary");parent.append(save);
    const table=element("dl",undefined,"node-properties");for(const [label,value] of [["节点编号",node.id],["类型",node.kind],["来源页数",node.sources.length],["审核状态",node.reviewStatus]]){table.append(element("dt",label),element("dd",value));}parent.append(table);
    parent.append(button("从此节点连线",()=>{connectFrom=node.id;message("请点击目标节点；Esc 取消连线。");}),button("新增关系",()=>showEdge(null,node.id)));

    const roleSelect=field(parent,"教学角色","select","","editTeachingRole");
    selectOptions(roleSelect,[["pending","待判断"],["learning","教学知识点"],["detail","素材细节"],["context_dependent","不同页面用途不一"]],node.teachingRole);
    parent.append(button("确认教学角色",()=>saveNode(node,{teaching_role:roleSelect.value,role_review_status:roleSelect.value==="pending" ? "pending" : "approved",role_reviewed_at:new Date().toISOString()})));
    parent.append(element("h3","先学什么"));
    const prerequisites=state.graph.links.filter(e=>e.relation==="PREREQUISITE_OF" && e.target===node.id && e.reviewStatus!=="rejected");
    if(!prerequisites.length)parent.append(element("p","尚未记录前置关系；这不代表没有前置知识。","muted"));
    for(const e of prerequisites)parent.append(button(nodeById(e.source).name+(e.reviewStatus==="approved" ? "" : "（待确认）"),()=>choose(e.source),"text-button"));
    const pathBox=element("div");parent.append(button("查看已确认的补学顺序",()=>{pathBox.replaceChildren();const result=M.approvedStudyPath(state.graph,node.id);if(result.status==="cycle"){pathBox.append(element("p","已确认的前置关系存在循环，请先审核修正。","form-error"));return;}if(!result.order.length){pathBox.append(element("p","当前没有已确认的补学顺序，不能用运行依赖或课件页序代替。","muted"));return;}const list=element("ol",undefined,"study-path");for(const id of result.order.concat(node.id))list.append(element("li",nodeById(id).name));pathBox.append(list);}),pathBox);
    renderSources(details(parent,"来源证据（"+node.sources.length+"页）"),node.sources);
    if(node.pageObservations.length){const contexts=details(parent,"逐页语境与模型建议（待确认）");for(const o of node.pageObservations){const b=element("div",undefined,"evidence"),s=node.sources.find(s=>s.page_id===o.page_id);b.append(element("strong",s ? "第 "+s.page_no+" 页 · "+(o.observed_name || node.name) : o.observed_name || node.name));if(o.description)b.append(element("p",o.description));if(o.learning_objective)b.append(element("p","建议学习目标："+o.learning_objective));if(Array.isArray(o.aliases_suggested_in_context) && o.aliases_suggested_in_context.length)b.append(element("p","本页建议称呼（非全局同义）："+o.aliases_suggested_in_context.join("、"),"muted"));contexts.append(b);}}
    const relationPanel=details(parent,"相关关系"),relations=state.graph.links.filter(e=>(e.source===node.id || e.target===node.id) && !["HAS_EVIDENCE","CONTAINS","BELONGS_TO"].includes(e.relation));
    relationPanel.open=true;
    for(const e of relations){const box=element("div",undefined,"relation-item");box.append(element("p",nodeById(e.source).name+" — "+(labels[e.relation] || e.relation)+" → "+nodeById(e.target).name,"relation-title"),element("span",e.reviewStatus==="approved" ? "教师已确认" : e.reviewStatus==="rejected" ? "教师未采纳" : "待教师确认","badge"));
      box.dataset.relation=e.relation;if(e.description)details(box,"关系依据（展开查看）").append(element("p",e.description));
      const reviews=array(state.raw.semantic_priority_review && state.raw.semantic_priority_review.items);
      for(const review of reviews)if(new Set([nodeById(e.source).name].concat(array(properties(nodeById(e.source).raw).previous_names))).has(review.source) && new Set([nodeById(e.target).name].concat(array(properties(nodeById(e.target).raw).previous_names))).has(review.target) && review.relation===e.relation)box.append(element("p","开发复核提示（待教师判断）："+review.reason,"review-item"));
      const sourceList=properties(e.raw).sources;if(Array.isArray(sourceList) && sourceList.length)renderSources(details(box,"查看关系来源"),sourceList);
      const actions=element("div",undefined,"button-row");actions.append(button("修改类型或依据",()=>showEdge(e)),button("删除",()=>removeEdge(e),"danger"));box.append(actions);relationPanel.append(box);}
    parent.append(button("＋ 添加关系",()=>showEdge(null,node.id),"text-button"));
    const candidates=state.graph.hierarchyCandidates.filter(e=>e.source===node.id || e.target===node.id);
    if(candidates.length){const fold=details(parent,"自动归组候选（"+candidates.length+"，非事实层级）");for(const e of candidates){const box=element("div",undefined,"review-item");box.append(element("p",nodeById(e.source).name+" → "+nodeById(e.target).name),element("p",e.description || "共享证据或挂靠规则不直接证明概念包含关系。"));fold.append(box);}}
    const isolated=state.graph.reviewItems.filter(item=>item.record && [item.record.source,item.record.target].some(v=>referenceNames.has(v) || String(v)===node.id));
    if(isolated.length){const fold=details(parent,"隔离的问题记录（"+isolated.length+"）");for(const item of isolated){const b=element("div",undefined,"review-item");b.append(element("p",item.record.description || item.reason),element("p",item.reason==="negated_runtime_requirement" ? "说明否定了依赖，未作为正向“需要”关系使用。" : item.reason==="relation_explanation_type_conflict" ? "关系类型与说明中明确选择的类型不一致，已隔离待核对。" : item.reason,"muted"));renderSources(b,item.record.sources || []);fold.append(b);}}
    const synonym=state.raw.synonym_review && state.raw.synonym_review.items || [],related=array(synonym).filter(x=>referenceNames.has(x.left) || referenceNames.has(x.right));
    if(related.length){const fold=details(parent,"名称合并建议（"+related.length+"，待审核）"),names={same_proposal:"可能同义",distinct:"概念不同",related_not_synonym:"相关但不宜合并",uncertain:"证据不足"};fold.append(element("p","开发阶段离线建议，未自动合并，也未代替教师审核。","muted"));for(const item of related){const b=element("div",undefined,"evidence");b.append(element("strong",item.left+" / "+item.right+"："+names[item.decision]),element("p",item.reason));for(const ref of array(item.evidence_refs))b.append(element("p","第 "+(ref.page_no || "?")+" 页："+ref.quote,"muted"));fold.append(b);}}
    parent.append(element("h3","管理"),button("删除此知识点及其关系",async()=>{if(!confirm("确定删除“"+node.name+"”及其相关关系？"))return;try{if(state.local){const native=Array.isArray(state.raw.concepts),arr=native ? state.raw.concepts : state.raw.nodes,index=arr.indexOf(node.raw);if(index<0)throw Error("没有找到原始节点。");arr.splice(index,1);const linkedRaw=new Set(state.graph.links.concat(state.graph.hierarchyCandidates).filter(e=>e.source===node.id || e.target===node.id).map(e=>e.raw));for(const key of native ? ["relationships","hierarchy_candidates"] : ["links","hierarchy_candidates"])if(Array.isArray(state.raw[key]))state.raw[key]=state.raw[key].filter(e=>!linkedRaw.has(e));state.selected=null;localChanged();}else{await mutateAndReload("/api/graph/node/"+encodeURIComponent(node.id)+"?course_name="+encodeURIComponent(courseName),"DELETE",undefined,"知识点已删除。");}}catch(error){message("删除失败："+error.message,true);}},"danger"));
    const deleteNodeButton=Array.from(parent.children).find(e=>e.tagName==="BUTTON" && e.classList.contains("danger"));
    const propertiesPanel=details(parent,"节点编号与属性");propertiesPanel.append(table);
    const advanced=details(parent,"教学角色与补学信息");let current=roleSelect.closest("label");
    while(current){const next=current.nextSibling;advanced.append(current);if(current===pathBox)break;current=next;}
    const primaryActions=element("div",undefined,"button-row node-primary-actions");parent.insertBefore(primaryActions,save);primaryActions.append(save,deleteNodeButton);
    parent.insertBefore(relationPanel,primaryActions.nextSibling);parent.insertBefore(advanced,relationPanel.nextSibling);
    relationPanel.classList.add("primary-relations");relationPanel.querySelector("summary").textContent="相关关系（"+relations.length+"）";
    const toolbar=element("div",undefined,"button-row");toolbar.append(button("＋ 新增关系",()=>showEdge(null,node.id)),button("从当前节点连线",()=>{connectFrom=node.id;message("请点击目标节点；Esc 取消连线。");}));
    relationPanel.insertBefore(toolbar,relationPanel.querySelector("summary").nextSibling);
    if(relations.length>8){const groups=new Map();for(const box of Array.from(relationPanel.children).filter(e=>e.classList.contains("relation-item"))){const key=box.dataset.relation;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(box);}let first=true;for(const [type,boxes] of groups){const group=details(relationPanel,(labels[type]||type)+"（"+boxes.length+"）");group.open=first;first=false;group.append(...boxes);}}
    for(const b of Array.from(parent.children).filter(e=>e.tagName==="BUTTON"&&["从此节点连线","新增关系","＋ 添加关系"].includes(e.textContent)))b.remove();
    for(const heading of Array.from(parent.children).filter(e=>e.tagName==="H3"&&e.textContent==="管理"))heading.remove();

  }
  function download(blob,name) {const url=URL.createObjectURL(blob),a=element("a");a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);}
  function exportSvgMarkup() {
    if(!chart || !state.view || !state.view.nodes.length)throw Error("当前视图没有可导出的图谱。");
    const exported=chart.exportSVG({title:$("courseName").textContent+" · "+({overview:"课程概览",hierarchy:"概念层级",prerequisites:"前置学习"})[state.mode],subtitle:"模型建议与教师确认分别标注；虚线关系待确认"});
    const svg=new DOMParser().parseFromString(exported.markup,"image/svg+xml").documentElement;
    window.CourseExportMetadata.annotateSvg(svg);
    return {markup:new XMLSerializer().serializeToString(svg),width:exported.width,height:exported.height};
  }
  async function exportImage(kind) {
    try {const exported=exportSvgMarkup(),svg=exported.markup,base=($("courseName").textContent || "课程")+"_"+state.mode;
      if(kind==="svg"){download(new Blob([svg],{type:"image/svg+xml;charset=utf-8"}),base+".svg");message("已导出当前筛选图谱 SVG。");return;}
      const width=exported.width,height=exported.height,scale=Math.min(2,12000/Math.max(width,height),Math.sqrt(32000000/(width*height))),url=URL.createObjectURL(new Blob([svg],{type:"image/svg+xml;charset=utf-8"}));
      try {const image=new Image();image.src=url;await image.decode();const canvas=document.createElement("canvas");canvas.width=Math.max(1,Math.round(width*scale));canvas.height=Math.max(1,Math.round(height*scale));const ctx=canvas.getContext("2d");ctx.fillStyle="#ffffff";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);const blob=await new Promise(resolve=>canvas.toBlob(resolve,"image/png"));if(!blob)throw Error("图片生成失败，可按课件筛选后重试，或导出SVG。");download(await window.CourseExportMetadata.pngWithAuthor(blob),base+".png");message("已导出当前筛选图谱清晰 PNG。");}finally{URL.revokeObjectURL(url);}
    }catch(error){message("导出失败："+error.message,true);}
  }
  $("nodeForm").addEventListener("submit",async e=>{
    e.preventDefault();const name=$("newNodeName").value.trim(),description=$("newNodeDescription").value.trim();
    if(!name || state.graph.nodes.some(n=>n.name===name)){$("nodeDialogError").textContent="名称不能为空或与当前课程重复。";return;}
    try {if(state.local){const id="local_"+Date.now()+"_"+Math.random().toString(36).slice(2,7),record={id,name,description,type:"knowledge",importance_level:2,teaching_role:"pending",role_review_status:"pending",review_status:"pending",origin:"teacher_edit"};if(Array.isArray(state.raw.concepts)){record.canonical_name=name;delete record.name;state.raw.concepts.push(record);}else state.raw.nodes.push(record);localChanged(name);}else{if(!await mutateAndReload("/api/graph/node","POST",{course_name:courseName,name,description,type:"knowledge"},"已新增知识点。"))return;const n=state.graph.nodes.find(n=>n.name===name);if(n)choose(n.id);}$("nodeDialog").close();}
    catch(error){$("nodeDialogError").textContent=error.message;}
  });
  $("edgeForm").addEventListener("submit",async e=>{
    e.preventDefault();const source=$("edgeSource").value,target=$("edgeTarget").value,relation=$("edgeType").value,description=$("edgeDescription").value.trim(),old=state.edge,review=$("edgeApproved").checked ? "approved" : "pending";
    const fail=text=>{$("edgeDialogError").textContent=text;};
    if(!description || source===target)return fail("请选择不同概念，并填写关系依据。");
    if(state.graph.links.some(e=>e.id!==(old && old.id) && e.source===source && e.target===target && e.relation===relation))return fail("相同关系已存在。");
    if(relation==="PREREQUISITE_OF" && M.wouldCreatePrerequisiteCycle(state.graph,source,target,old && old.id))return fail("这条先修关系会形成循环，未保存。请先审核相关关系。");
    try {const patch={relation,description,review_status:review,review_required:review!=="approved",origin:"teacher_edit",reviewed_at:review==="approved" ? new Date().toISOString() : null,auto_generated:false};
      if(state.local){const native=Array.isArray(state.raw.concepts);if(old){const found=edgeRecord(old);if(found.index<0)throw Error("没有找到原始关系。");writeFields(found.records[found.index],Object.assign({},patch,native ? {} : {value:relation}));}else{const key=native ? "relationships" : "links";state.raw[key] ||= [];state.raw[key].push(Object.assign({id:"local_edge_"+Date.now(),source:native ? nodeById(source).name : source,target:native ? nodeById(target).name : target},patch,native ? {} : {value:relation}));}localChanged();}
      else {if(!await mutateAndReload("/api/graph/link",old ? "PUT" : "POST",Object.assign({course_name:courseName,source_id:source,target_id:target,old_relation:old ? old.relation : undefined,relation_id:old ? (old.raw.relation_id || old.raw.id || old.raw.relationship_id) : undefined},patch),review==="approved" ? "关系已由你明确确认并保存。" : "关系已保存，仍待确认。"))return;}
      $("edgeDialog").close();
    } catch(error){fail(error.message);}
  });
  $("toolbarAddNode").onclick=()=>$("addNode").click();
  $("toolbarAddEdge").onclick=()=>showEdge(null,state.selected);
  $("deleteSelectedEdge").onclick=async()=>{if(state.edge){await removeEdge(state.edge);$("edgeDialog").close();}};
  $("relayoutGraph").onclick=()=>chart?.relayout();
  $("pauseGraph").onclick=()=>{$("pauseGraph").textContent=chart?.pause()?"恢复牵动":"暂停牵动";};
  $("pinNode").onclick=()=>{const result=chart?.pin();message(result===null?"请先选择节点。":result?"节点位置已固定，右键可解锁。":"节点已解锁。");};
  $("undoGraph").onclick=async()=>{if(state.local)return message("导入副本请重新导入原文件恢复；撤销用于已保存的课程。",true);if(!confirm("撤销当前课程上一次保存的图谱修改？"))return;try{await mutateAndReload("/api/graph/undo","POST",{course_name:courseName},"已撤销上一次修改。");}catch(e){message(e.message,true);}};
  $("addNode").onclick=()=>{if(!state.graph)return message("请先加载或导入图谱。",true);$("nodeForm").reset();$("nodeDialogError").textContent="";$("nodeDialog").showModal();};
  document.querySelectorAll("[data-close]").forEach(b=>b.onclick=()=>$(b.dataset.close).close());
  document.querySelectorAll("[data-mode]").forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;render();});
  ["graphSearch","documentFilter","importanceFilter","showDetails","showPending"].forEach(id=>$(id).addEventListener(id==="graphSearch" ? "input" : "change",()=>{render();}));
  $("importGraph").onclick=()=>{++requestVersion;$("reloadGraph").disabled=false;if(state.loadPhase==="loading"){state.loadPhase=state.graph.nodes.length ? "ready" : "idle";renderEmpty();}$("graphFile").click();};
  $("graphFile").onchange=async e=>{const file=e.target.files[0],version=requestVersion;if(!file)return;try{const raw=JSON.parse(await file.text());M.normalize(raw);if(version!==requestVersion)return;if(state.dirty && !confirm("当前有未导出的修改，仍要导入新图谱吗？"))return;state.selected=null;state.mode="overview";state.importLabel=file.name;$("graphSearch").value="";$("documentFilter").value="";apply(raw,{local:true,dirty:false,ready:true});message("已导入 "+file.name+"。当前为本地副本，可编辑并导出保存。");}catch(error){if(version===requestVersion)message("导入失败："+error.message,true);}finally{e.target.value="";}};
  $("reloadGraph").onclick=loadGraph;
  $("exportJson").onclick=()=>{if(!state.raw)return message("请先加载或导入图谱。",true);download(new Blob([JSON.stringify(state.raw,null,2)],{type:"application/json;charset=utf-8"}),($("courseName").textContent || "课程")+"_图谱.json");message("已生成图谱JSON下载，请确认文件已保存。");};
  $("exportSvg").onclick=()=>exportImage("svg");$("exportPng").onclick=()=>exportImage("png");$("fitGraph").onclick=()=>{if(chart)chart.fit();};
  $("zoomOut").onclick=()=>{if(chart)chart.zoomBy(1/1.25);};
  $("zoomIn").onclick=()=>{if(chart)chart.zoomBy(1.25);};
  $("toggleInspector").onclick=()=>setInspector($("toggleInspector").getAttribute("aria-expanded")!=="true");
  $("closeInspector").onclick=()=>{setInspector(false);$("toggleInspector").focus();};
  $("toggleFilters").onclick=()=>setFilters($("filterPanel").hidden);
  $("toggleFullscreen").onclick=()=>{const active=document.querySelector(".kg-page").classList.toggle("is-fullscreen");$("toggleFullscreen").setAttribute("aria-pressed",String(active));$("toggleFullscreen").textContent=active ? "退出聚焦" : "聚焦";};
  document.addEventListener("keydown",event=>{if(event.key==="Escape")connectFrom=null;if(event.key==="Escape" && !document.querySelector("dialog[open]")){setFilters(false);setInspector(false);}});
  $("zoomReset").onclick=()=>{if(chart)chart.resetZoom();};
  $("showEdgeLabels").onchange=()=>{if(chart)chart.setEdgeLabels($("showEdgeLabels").checked);};
  window.addEventListener("beforeunload",e=>{if(state.dirty){e.preventDefault();e.returnValue="";}});
  if(typeof ResizeObserver!=="undefined"){const fieldResize=new ResizeObserver(()=>{requestAnimationFrame(()=>autosize($("editNodeDescription")));});fieldResize.observe($("inspector"));}
  if(!M){message("图谱数据模块加载失败，请检查静态文件。",true);return;}
  if(window.cytoscape && window.CourseGraphCanvas && window.CourseGraphMapLayout){chart=window.CourseGraphCanvas.create($("graphContainer"),{cytoscape:window.cytoscape,model:M,mapLayout:window.CourseGraphMapLayout,onVisible,onSelect:id=>choose(id,false,true),onEdgeSelect:id=>{const edge=state.graph.links.find(e=>e.id===id);if(edge)showEdge(edge);},onZoom:zoom=>{$("zoomLevel").textContent=Math.round(zoom*100)+"%";}});}
  else message("图谱绘制组件加载失败，知识点列表与编辑仍可使用。",true);
  const bootstrap=window.__COURSE_GRAPH_BOOTSTRAP__;
  if(bootstrap){apply(bootstrap,{local:true,dirty:false,ready:true});message("已打开图谱副本，可编辑并导出保存。");}
  else {apply({nodes:[],links:[]},{local:false,dirty:false});loadGraph();}
}());

