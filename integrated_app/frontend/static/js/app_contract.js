/* Shared same-origin HTTP and task-state contracts. */
(function (root, factory) {
  "use strict";
  const value = factory();
  if (typeof module === "object" && module.exports) module.exports = value;
  else root.CourseApp = value;
})(typeof window !== "undefined" ? window : this, function () {
  "use strict";
  const terminal = new Set(["succeeded", "failed", "cancelled"]);
  const labels = {queued: "等待处理", running: "处理中", succeeded: "处理成功", failed: "处理失败", cancelled: "已取消", cancel_requested: "正在取消"};
  const stages = {queued: "等待处理", validating: "检查文件", extraction: "提取课件", extracting: "提取课件", graph: "生成知识图谱", building: "生成知识图谱", database: "保存数据库", storing: "保存数据库", exporting: "发布课程图谱", publishing: "发布课程图谱", completed: "处理完成"};
  function courseQuery(course) {
    const params = new URLSearchParams();
    if (course.courseId) params.set("course_id", course.courseId);
    if (course.courseName) params.set("course_name", course.courseName);
    return params.toString();
  }
  function graphUrl(course) { return "/knowledge-graph/?" + courseQuery(course); }
  function localSourceUrl(value) {
    return typeof value === "string" && /^\/api\/jobs\/job_[a-f0-9]{32}\/source(?:#page=[1-9][0-9]*)?$/.test(value) ? value : null;
  }
  function localImageUrl(value) {
    if (typeof value !== "string" || /[\\?#]/.test(value)) return null;
    const match = /^\/api\/jobs\/job_[a-f0-9]{32}\/assets\/(.+)$/.exec(value);
    if (!match) return null;
    let relative; try { relative = decodeURIComponent(match[1]); } catch (_) { return null; }
    if (/[\\%\u0000-\u001f]/.test(relative) || relative.split("/").some(part => !part || part === "." || part === "..") || !/\.(png|jpe?g)$/i.test(relative)) return null;
    return value;
  }
  function jobsForCourse(payload, course) {
    const jobs = Array.isArray(payload) ? payload : (payload && Array.isArray(payload.jobs) ? payload.jobs : []);
    return jobs.filter(job => course.courseId ? String(job.course_id || "") === String(course.courseId) : Boolean(course.courseName) && job.course_name === course.courseName);
  }
  function jobPresentation(job) {
    const status = String(job.status || "queued");
    let progress = typeof job.progress === "number" && Number.isFinite(job.progress) ? Math.max(0, Math.min(100, job.progress)) : null;
    if (status === "succeeded") progress = 100;
    else if (progress !== null) progress = Math.min(progress, 99);
    return {status, label: labels[status] || "状态待确认", stage: stages[job.stage] || String(job.stage || ""), progress, terminal: terminal.has(status), failed: status === "failed", error: typeof job.error === "string" ? job.error : job.error && (job.error.message || job.error.code) || ""};
  }
  async function request(path, options) {
    if (typeof path !== "string" || !path.startsWith("/") || path.startsWith("//")) throw Error("接口地址必须使用本站路径");
    const settings = Object.assign({}, options || {}, {credentials: "same-origin"});
    if (Object.prototype.hasOwnProperty.call(settings, "json")) {
      settings.headers = Object.assign({}, settings.headers || {}, {"Content-Type": "application/json"});
      settings.body = JSON.stringify(settings.json); delete settings.json;
    }
    const response = await fetch(path, settings);
    let data; try { data = await response.json(); } catch (_) { throw Error("服务返回格式不正确，请稍后重试"); }
    if (!response.ok || data && data.success === false) {
      const error = Error(data && (data.error && (data.error.message || data.error) || data.message) || "请求失败（" + response.status + "）");
      error.status = response.status; throw error;
    }
    return data;
  }
  return {request, courseQuery, graphUrl, jobsForCourse, jobPresentation, localSourceUrl, localImageUrl};
});
