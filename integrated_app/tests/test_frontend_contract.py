"""Executable JS checks for course isolation and honest task/error presentation."""
from pathlib import Path
import base64
import io
import json
from html.parser import HTMLParser
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
JAVASCRIPT = ROOT / "frontend/static/js"
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node.js is required for frontend contract checks")
class FrontendContractTests(unittest.TestCase):
    def js(self, code):
        program = "const assert = require('node:assert/strict');\nconst A = require(" + repr(str(JAVASCRIPT / "app_contract.js")) + ");\n" + code
        result = subprocess.run([NODE, "-e", program], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_identical_course_names_do_not_mix_jobs(self):
        self.js("""
const jobs = [{job_id:'one',course_id:'course-a',course_name:'同名课程',status:'failed'},
              {job_id:'two',course_id:'course-b',course_name:'同名课程',status:'running'},
              {job_id:'unscoped',course_name:'同名课程',status:'succeeded'}];
assert.deepEqual(A.jobsForCourse({jobs},{courseId:'course-a',courseName:'同名课程'}).map(x=>x.job_id),['one']);
assert.deepEqual(A.jobsForCourse({jobs},{courseId:'missing',courseName:'同名课程'}),[]);
assert.deepEqual(A.jobsForCourse({jobs},{}),[]);
""")

    def test_failure_and_requested_cancel_cannot_display_success(self):
        self.js("""
const failed=A.jobPresentation({status:'failed',progress:100,error:{message:'提取失败'}});
assert.equal(failed.failed,true);assert.equal(failed.label,'处理失败');assert.equal(failed.error,'提取失败');assert.ok(failed.progress<100);
const cancelling=A.jobPresentation({status:'cancel_requested',progress:100});
assert.equal(cancelling.terminal,false);assert.equal(cancelling.label,'正在取消');assert.ok(cancelling.progress<100);
const complete=A.jobPresentation({status:'succeeded',progress:20});
assert.equal(complete.terminal,true);assert.equal(complete.progress,100);
assert.equal(A.jobPresentation({status:'running'}).progress,null);
""")

    def test_requests_keep_same_origin_session_and_surface_server_errors(self):
        self.js("""
(async()=>{
let observed;
global.fetch=async(path,settings)=>{observed={path,settings};return {ok:true,status:200,json:async()=>({success:true})};};
await A.request('/api/upload',{method:'POST',json:{course_id:'course-a'}});
assert.equal(observed.path,'/api/upload');assert.equal(observed.settings.credentials,'same-origin');
assert.deepEqual(JSON.parse(observed.settings.body),{course_id:'course-a'});
await assert.rejects(A.request('https://another.example/api'),/本站路径/);
await assert.rejects(A.request('//another.example/api'),/本站路径/);
global.fetch=async()=>({ok:false,status:503,json:async()=>({success:false,error:'课程检索尚未配置'})});
await assert.rejects(A.request('/api/qa/'),/课程检索尚未配置/);
global.fetch=async()=>({ok:true,status:200,json:async()=>({success:false,error:'任务失败'})});
await assert.rejects(A.request('/api/jobs'),/任务失败/);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")

    def test_course_navigation_preserves_identity_and_escapes_names(self):
        self.js("""
const url=new URL(A.graphUrl({courseId:'id/a',courseName:'课程 & <符号>'}),'http://127.0.0.1');
assert.equal(url.pathname,'/knowledge-graph/');assert.equal(url.searchParams.get('course_id'),'id/a');
assert.equal(url.searchParams.get('course_name'),'课程 & <符号>');
""")

    def test_png_export_has_unicode_author_and_preserves_image_pixels(self):
        from PIL import Image
        source = io.BytesIO()
        Image.new("RGB", (3, 2), (12, 34, 56)).save(source, format="PNG")
        script = "const M=require(" + repr(str(JAVASCRIPT / "export_metadata.js")) + ");\n"
        script += "(async()=>{const input=new Blob([Buffer.from(" + json.dumps(base64.b64encode(source.getvalue()).decode())
        script += ",'base64')],{type:'image/png'});const output=await M.pngWithAuthor(input);process.stdout.write(Buffer.from(await output.arrayBuffer()).toString('base64'));})().catch(e=>{console.error(e);process.exitCode=1;});"
        result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(io.BytesIO(base64.b64decode(result.stdout))) as image:
            self.assertEqual(image.info["Author"], "昕煜 冷")
            self.assertEqual(image.size, (3, 2))
            self.assertEqual(image.getpixel((1, 1)), (12, 34, 56))

    def test_evidence_pdf_links_follow_each_document_job_not_latest_job(self):
        self.js("const M=require(" + repr(str(JAVASCRIPT / "knowledge_graph_model.js")) + ");\n" + """
const older='job_'+'a'.repeat(32), newer='job_'+'b'.repeat(32);
const raw={nodes:[{labels:['Document'],properties:{id:'doc-one',document_id:'doc-one',job_id:older}},
 {labels:['Page'],properties:{id:'page-one',page_id:'page-one',document_id:'doc-one',page_no:4,job_id:older}},
 {labels:['Document'],properties:{id:'doc-two',document_id:'doc-two',job_id:newer}},
 {labels:['Page'],properties:{id:'page-two',page_id:'page-two',document_id:'doc-two',page_no:7,job_id:newer}}],
 integration:{job_id:newer,source_url:'/api/jobs/'+newer+'/source'}};
assert.equal(M.sourcePdfLink(raw,{page_id:'page-one',document_id:'doc-one',page_no:4}).href,'/api/jobs/'+older+'/source#page=4');
assert.equal(M.sourcePdfLink(raw,{page_id:'page-two',document_id:'doc-two',page_no:7}).href,'/api/jobs/'+newer+'/source#page=7');
assert.equal(M.sourcePdfLink(raw,{page_id:'page-one',document_id:'doc-two',page_no:4}),null);
assert.equal(M.sourcePdfLink(raw,{page_id:'page-one',document_id:'doc-one',page_no:9}),null);
const noJobs=JSON.parse(JSON.stringify(raw));noJobs.nodes.forEach(node=>delete node.properties.job_id);
assert.equal(M.sourcePdfLink(noJobs,{page_id:'page-one',document_id:'doc-one',page_no:4}),null);
const single={documents:[{document_id:'doc-one'}],integration:{job_id:older,source_url:'/api/jobs/'+older+'/source'}};
assert.equal(M.sourcePdfLink(single,{document_id:'doc-one',page_no:4}).href,'/api/jobs/'+older+'/source#page=4');
assert.equal(M.sourcePdfLink(single,{document_id:'another-document',page_no:4}),null);
single.integration.source_url='https://external.example/api/jobs/'+older+'/source';
assert.equal(M.sourcePdfLink(single,{document_id:'doc-one',page_no:4}),null);
single.integration.source_url='file:///api/jobs/'+older+'/source';
assert.equal(M.sourcePdfLink(single,{document_id:'doc-one',page_no:4}),null);
""")

    def test_material_source_urls_are_job_scoped_and_cannot_escape_assets(self):
        self.js("""
const job='job_'+'c'.repeat(32),base='/api/jobs/'+job;
assert.equal(A.localSourceUrl(base+'/source#page=5'),base+'/source#page=5');
assert.equal(A.localSourceUrl('https://external.example'+base+'/source'),null);
assert.equal(A.localSourceUrl('file://'+base+'/source'),null);
assert.equal(A.localSourceUrl(base+'/source#page=0'),null);
assert.equal(A.localImageUrl(base+'/assets/figure_1.png'),base+'/assets/figure_1.png');
assert.equal(A.localImageUrl(base+'/assets/subfolder/figure_2.jpg'),base+'/assets/subfolder/figure_2.jpg');
for(const tail of ['../secret.png','%2e%2e/secret.png','%252e%252e/secret.png','image.svg','image.png?path=outside'])assert.equal(A.localImageUrl(base+'/assets/'+tail),null);
assert.equal(A.localImageUrl('https://external.example'+base+'/assets/image.png'),null);
""")

    def test_auth_controls_have_bound_accessible_labels(self):
        class Controls(HTMLParser):
            def __init__(self):
                super().__init__()
                self.controls = set()
                self.labels = set()

            def handle_starttag(self, tag, attributes):
                attributes = dict(attributes)
                if tag in {"input", "select"} and attributes.get("id"):
                    self.controls.add(attributes["id"])
                if tag == "label" and attributes.get("for"):
                    self.labels.add(attributes["for"])

        for filename in ("login.html", "register.html"):
            parser = Controls()
            parser.feed((ROOT / "frontend/frontend/templates/frontend" / filename).read_text(encoding="utf-8"))
            self.assertTrue(parser.controls)
            self.assertEqual(parser.controls - parser.labels, set(), filename)


if __name__ == "__main__":
    unittest.main()
