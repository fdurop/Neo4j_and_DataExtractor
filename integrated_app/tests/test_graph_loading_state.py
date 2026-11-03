"""Exercise the real controller's async load states without paid services or a browser."""
from pathlib import Path
import json
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node.js is required for graph controller tests")
class GraphLoadingStateTests(unittest.TestCase):
    def js(self, code):
        prelude = "const ROOT=" + json.dumps(str(ROOT)) + ";\n" + r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const M=require(path.join(ROOT,'frontend/static/js/knowledge_graph_model.js'));
const L=require(path.join(ROOT,'frontend/static/js/graph_map_layout.js'));
const sample={nodes:[{id:'one',name:'Example concept',type:'knowledge',importance_level:1,
 suggested_teaching_role:'learning',review_status:'pending'}],links:[]};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
async function settle(){await tick();await tick();}
function boot(options={}){
 const ids=new Map(),pending=[],events={};
 class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.dataset={};this.attributes={};this.value='';this.checked=false;this.hidden=false;this.textContent='';this.listeners={};this.style={};
   this.classList={toggle(){},add(){},remove(){}};}
  removeAttribute(key){delete this.attributes[key];}
  set id(value){this._id=value;ids.set(value,this);} get id(){return this._id;}
  append(...children){for(const child of children){this.children.push(child);child.parent=this;}}
  replaceChildren(...children){this.children=[];this.append(...children);}
  setAttribute(name,value){this.attributes[name]=String(value);} getAttribute(name){return this.attributes[name] || null;}
  addEventListener(name,fn){this.listeners[name]=fn;}
  fire(name,event={}){return (this.listeners[name] || this['on'+name] || (()=>{}))(event);}
  querySelectorAll(){return [];}
  remove(){ids.delete(this.id);if(this.parent)this.parent.children=this.parent.children.filter(item=>item!==this);}
  focus(){} click(){return this.fire('click');} reset(){} showModal(){this.open=true;} close(){this.open=false;}
 }
 const template=fs.readFileSync(path.join(ROOT,'frontend/frontend/templates/frontend/knowledge_graph.html'),'utf8');
 for(const match of template.matchAll(/\bid="([^"]+)"/g)){const element=new Element();element.id=match[1];}
 ids.get('importanceFilter').value='3';ids.get('showPending').checked=true;ids.get('filterPanel').hidden=true;
 const workspace=new Element(),page=new Element(),modes=['overview','hierarchy','prerequisites'].map(mode=>{const e=new Element('button');e.dataset.mode=mode;return e;});
 const document={getElementById:id=>ids.get(id)||null,createElement:tag=>new Element(tag),
  querySelector:s=>s==='.workspace' ? workspace : s==='.kg-page' ? page : null,
  querySelectorAll:s=>s==='[data-mode]' ? modes : [],addEventListener:(name,fn)=>events[name]=fn};
 const window={CourseGraphModel:M,CourseGraphMapLayout:L,cytoscape(){},addEventListener(){},
  CourseGraphCanvas:{create(){return {render(){},select(){},fit(){},zoomBy(){},resetZoom(){},setEdgeLabels(){}};}}};
 if(options.bootstrap)window.__COURSE_GRAPH_BOOTSTRAP__=options.bootstrap;
 const location={search:options.query===undefined ? '?course_id=course-a&course_name=Course' : options.query,origin:'http://127.0.0.1:8769'};
 const context={window,document,location,URL,URLSearchParams,console,setTimeout,clearTimeout,confirm:()=>true,
  fetch:(url,settings)=>new Promise(resolve=>pending.push({url,settings,resolve}))};
 vm.runInNewContext(fs.readFileSync(path.join(ROOT,'frontend/static/js/knowledge_graph.js'),'utf8'),context);
 return {ids,pending,modes,respond(index,status,body){pending[index].resolve({ok:status>=200&&status<300,status,json:async()=>body});}};
}
"""
        result = subprocess.run([NODE, "-"], input=prelude + "\n(async()=>{\n" + code + "\n})().catch(error=>{console.error(error);process.exitCode=1;});",
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_loading_is_not_reported_as_a_filter_failure(self):
        self.js("""
const app=boot();
assert.equal(app.pending.length,1);
assert.equal(app.ids.get('emptyTitle').textContent,'正在加载课程图谱');
assert.equal(app.ids.get('graphContainer').getAttribute('aria-busy'),'true');
assert.equal(app.ids.get('reloadGraph').disabled,true);
assert.equal(app.pending[0].settings.credentials,'same-origin');
app.respond(0,200,sample);await settle();
assert.equal(app.ids.get('graphEmpty').hidden,true);
assert.equal(app.ids.get('graphContainer').getAttribute('aria-busy'),'false');
assert.equal(app.ids.get('reloadGraph').disabled,false);
""")

    def test_unauthorized_response_explains_login_and_offers_a_real_link(self):
        self.js("""
const app=boot();app.respond(0,401,{success:false,error:'请先登录'});await settle();
assert.equal(app.ids.get('emptyTitle').textContent,'请先登录后查看课程图谱');
assert.equal(app.ids.get('emptyAction').href,'/login/');
assert.equal(app.ids.get('emptyAction').textContent,'前往登录');
assert.equal(app.ids.get('graphContainer').getAttribute('aria-busy'),'false');
assert.equal(app.ids.get('graphEmpty').hidden,false);
""")

    def test_database_failure_has_retry_and_an_empty_success_is_distinct(self):
        self.js("""
const app=boot();app.respond(0,503,{success:false,error:'数据库图谱读取失败'});await settle();
assert.equal(app.ids.get('emptyTitle').textContent,'课程图谱加载失败');
assert.match(app.ids.get('emptyDescription').textContent,/数据库/);
app.ids.get('emptyAction').click();
assert.equal(app.ids.get('emptyTitle').textContent,'正在加载课程图谱');
app.respond(1,200,{nodes:[],links:[]});await settle();
assert.equal(app.ids.get('emptyTitle').textContent,'课程还没有知识点');
assert.equal(app.ids.get('emptyAction').href,'/teacher/');
""")

    def test_actual_filter_empty_can_be_cleared_without_refetching(self):
        self.js("""
const app=boot();app.respond(0,200,sample);await settle();
const search=app.ids.get('graphSearch');search.value='no matching concept';search.fire('input');
assert.equal(app.ids.get('emptyTitle').textContent,'没有符合筛选的知识点');
assert.equal(app.ids.get('emptyAction').textContent,'清空筛选');
app.ids.get('emptyAction').click();
assert.equal(app.ids.get('graphEmpty').hidden,true);
assert.equal(search.value,'');assert.equal(app.pending.length,1);
""")

    def test_older_success_cannot_replace_a_newer_unauthorized_request(self):
        self.js("""
const app=boot();app.ids.get('reloadGraph').click();
app.respond(0,200,sample);await settle();
assert.equal(app.ids.get('emptyTitle').textContent,'正在加载课程图谱');
app.respond(1,401,{success:false,error:'请先登录'});await settle();
assert.equal(app.ids.get('emptyTitle').textContent,'请先登录后查看课程图谱');
""")

    def test_local_import_recovers_from_login_error_without_database_access(self):
        self.js("""
const app=boot();app.respond(0,401,{success:false,error:'请先登录'});await settle();
const input=app.ids.get('graphFile');input.files=[{name:'saved.json',text:async()=>JSON.stringify(sample)}];
await input.onchange({target:input});
assert.equal(app.ids.get('graphEmpty').hidden,true);
assert.equal(app.pending.length,1);
assert.match(app.ids.get('workspaceNote').textContent,/导入副本/);
const local=boot({bootstrap:sample});
assert.equal(local.ids.get('graphEmpty').hidden,true);assert.equal(local.pending.length,0);
""")

    def test_course_id_alone_is_sufficient_to_start_loading(self):
        self.js("""
const app=boot({query:'?course_id=course-a'});
assert.equal(app.pending.length,1);assert.match(app.pending[0].url,/course_id=course-a/);
app.respond(0,200,sample);await settle();assert.equal(app.ids.get('graphEmpty').hidden,true);
""")

    def test_mutation_blocks_until_write_and_refreshed_graph_are_complete(self):
        self.js("""
const app=boot();app.respond(0,200,sample);await settle();
const action=app.ids.get('undoGraph').click();await settle();
assert.equal(app.ids.get('graphBusyDialog').open,true);
assert.equal(app.ids.get('graphBusySpinner').hidden,false);
assert.match(app.ids.get('graphBusyTitle').textContent,/撤销/);
app.ids.get('undoGraph').click();await settle();assert.equal(app.pending.length,2);
let cancelled=false;app.ids.get('graphBusyDialog').fire('cancel',{preventDefault(){cancelled=true;}});assert(cancelled);
app.respond(1,200,{success:true,review_revision:2});await settle();
assert.equal(app.pending.length,3);assert.equal(app.ids.get('graphBusyDialog').open,true);
assert.match(app.ids.get('graphBusyTitle').textContent,/更新图谱/);
app.respond(2,200,{nodes:[],links:[],review_revision:2});await settle();await action;
assert.equal(app.ids.get('graphBusyDialog').open,false);
""")

    def test_failure_is_visible_and_retry_only_reloads_without_repeating_write(self):
        self.js("""
const app=boot();app.respond(0,200,sample);await settle();
const action=app.ids.get('undoGraph').click();await settle();
app.respond(1,503,{error:'模拟网络失败'});await settle();await action;
assert.equal(app.ids.get('graphBusyDialog').open,true);
assert.equal(app.ids.get('graphBusySpinner').hidden,true);
assert.equal(app.ids.get('graphBusyActions').hidden,false);
assert.match(app.ids.get('graphBusyText').textContent,/模拟网络失败/);
const retry=app.ids.get('graphBusyRetry').click();await settle();
assert.equal(app.pending.length,3);assert.equal(app.pending[2].settings.method,'GET');
app.respond(2,200,sample);await settle();await retry;
assert.equal(app.ids.get('graphBusyDialog').open,false);
assert.equal(app.pending.filter(r=>r.settings.method==='POST').length,1);
""")


if __name__ == '__main__':
    unittest.main()
