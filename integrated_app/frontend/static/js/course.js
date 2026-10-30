(function () {
  "use strict";
  const A = window.CourseApp;
  const params = new URLSearchParams(window.location.search);
  const course = {courseId: params.get("course_id") || "", courseName: params.get("course_name") || ""};
  const $ = id => document.getElementById(id);
  let polling = false, previousJobs = new Map();
  function showStatus(message, error) {
    const box = $("statusMessage"); box.textContent = message;
    box.className = "status-message " + (error ? "error" : "success"); box.hidden = false;
  }
  function text(tag, value, cls) {
    const element = document.createElement(tag); element.textContent = value;
    if (cls) element.className = cls; return element;
  }
  async function uploadFile(file) {
    if (!course.courseId) return showStatus("请从课程列表选择课程后上传", true);
    if (!file.name.toLowerCase().endsWith(".pdf")) return showStatus("当前支持上传 PDF 课件", true);
    $("uploadProgress").hidden = false; $("progressFill").style.width = "0%";
    $("progressText").textContent = "正在上传文件……";
    const body = new FormData(); body.append("file", file); body.append("course_id", course.courseId); body.append("course_name", course.courseName);
    try {
      const result = await A.request("/api/upload", {method: "POST", body});
      if (!result.job_id) throw Error("上传已返回，但缺少任务编号，请检查处理状态");
      $("progressFill").style.width = "100%"; $("progressText").textContent = "文件上传完成，后台处理状态见下方";
      showStatus("已创建处理任务。提取、建图和保存完成后才会显示“处理成功”。");
      await Promise.all([loadFiles(), loadJobs()]);
    } catch (error) { $("progressText").textContent = "上传失败"; showStatus(error.message, true); }
  }
  async function loadFiles() {
    try {
      const data = await A.request("/api/files?" + A.courseQuery(course)), items = $("fileItems"); items.replaceChildren();
      const files = Array.isArray(data.files) ? data.files : [];
      if (!files.length) return items.append(text("div", "暂无文件", "no-files"));
      for (const file of files) {
        const row = document.createElement("div"); row.className = "file-item";
        const info = text("div", file.original_filename || file.filename || "未命名文件", "file-info");
        const actions = document.createElement("div"); actions.className = "file-actions";
        const remove = text("button", "删除"); remove.type = "button";
        remove.addEventListener("click", async () => {
          if (!confirm("确定从上传列表移除这个文件？处理产物会保留。")) return;
          try { await A.request("/api/files/" + encodeURIComponent(file.id || file.job_id) + "?" + A.courseQuery(course), {method: "DELETE"}); showStatus("文件已从列表移除"); await loadFiles(); }
          catch (error) { showStatus(error.message, true); }
        });
        actions.append(remove); row.append(info, actions); items.append(row);
      }
    } catch (error) { showStatus("文件列表加载失败：" + error.message, true); }
  }
  function renderJobs(jobs) {
    const container = $("processingItems"); container.replaceChildren(); $("processingProgress").hidden = !jobs.length;
    for (const job of jobs) {
      const state = A.jobPresentation(job), row = document.createElement("article"); row.className = "processing-item task-" + state.status; row.dataset.jobId = job.job_id || job.id || "";
      row.append(text("strong", job.original_filename || job.filename || "处理任务"));
      row.append(text("p", state.label + (state.stage && !state.terminal ? " · " + state.stage : ""), "task-state"));
      if (state.progress !== null) {
        const progress = document.createElement("progress"); progress.max = 100; progress.value = state.progress; progress.setAttribute("aria-label", "处理进度"); row.append(progress); row.append(text("span", " " + Math.round(state.progress) + "%"));
      }
      if (job.message && !state.failed) row.append(text("p", job.message, "task-message"));
      if (state.failed) row.append(text("p", state.error || "处理失败，请查看任务记录后重试", "task-error"));
      if (state.status === "succeeded") {
        const link = text("a", "查看生成的知识图谱"); link.href = A.graphUrl(course); row.append(link);
      } else if (!state.terminal && state.status !== "cancel_requested") {
        const cancel = text("button", "取消任务"); cancel.type = "button";
        cancel.addEventListener("click", async () => {
          cancel.disabled = true;
          try { await A.request("/api/jobs/" + encodeURIComponent(job.job_id || job.id) + "/cancel", {method: "POST"}); await loadJobs(); }
          catch (error) { cancel.disabled = false; showStatus(error.message, true); }
        }); row.append(cancel);
      }
      container.append(row);
    }
  }
  async function loadJobs() {
    if (polling) return; polling = true;
    try {
      const payload = await A.request("/api/jobs?" + A.courseQuery(course)), jobs = A.jobsForCourse(payload, course);
      for (const job of jobs) { const id = job.job_id || job.id; if (job.status === "succeeded" && previousJobs.get(id) && previousJobs.get(id) !== "succeeded") showStatus("课件处理成功，可以查看知识图谱"); }
      previousJobs = new Map(jobs.map(job => [job.job_id || job.id, job.status])); renderJobs(jobs);
    } catch (error) { showStatus("任务状态读取失败：" + error.message, true); }
    finally { polling = false; }
  }
  window.navigateToKnowledgeGraph = () => { window.location.href = A.graphUrl(course); };
  window.addEventListener("DOMContentLoaded", () => {
    if (!course.courseId) { $("chooseFileBtn").disabled = true; showStatus("未选择课程，请返回课程列表", true); return; }
    $("courseTitle").textContent = course.courseName || "课程资料";
    $("materialReviewLink").href = "/repair/?" + A.courseQuery(course);
    $("chooseFileBtn").addEventListener("click", () => $("fileInput").click());
    $("fileInput").addEventListener("change", event => { for (const file of event.target.files) uploadFile(file); event.target.value = ""; });
    const area = $("uploadArea");
    area.addEventListener("dragover", event => { event.preventDefault(); area.classList.add("drag-over"); });
    area.addEventListener("dragleave", () => area.classList.remove("drag-over"));
    area.addEventListener("drop", event => { event.preventDefault(); area.classList.remove("drag-over"); for (const file of event.dataTransfer.files) uploadFile(file); });
    loadFiles(); loadJobs(); let timer = setInterval(loadJobs, 2000);
    window.addEventListener("pagehide", () => clearInterval(timer));
    window.addEventListener("pageshow", event => { if (event.persisted) { loadFiles(); loadJobs(); timer = setInterval(loadJobs, 2000); } });
  });
})();
