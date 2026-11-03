(function () {
  "use strict";
  const A = window.CourseApp;
  async function loadCourses() {
    const container = document.getElementById("courseList"); if (!container) return;
    try {
      const result = await A.request("/api/courses"), courses = Array.isArray(result) ? result : result.courses || result.data || []; container.replaceChildren();
      if (!courses.length) { container.textContent = "暂无课程，请新建课程后上传课件。"; return; }
      for (const course of courses) {
        const id = course.course_id || course.id, name = course.course_name || course.name;
        const row = document.createElement("div"); row.className = "course-item";
        const info = document.createElement("div"); info.className = "course-info";
        const link = document.createElement("a"); link.className = "course-name"; link.textContent = name; link.href = "/course/?" + A.courseQuery({courseId: id, courseName: name});
        const graph = document.createElement("a"); graph.textContent = "查看图谱"; graph.className = "course-graph-link"; graph.href = A.graphUrl({courseId: id, courseName: name}); info.append(link, graph);
        const actions = document.createElement("div"); actions.className = "course-actions";
        const remove = document.createElement("button"); remove.className = "delete-btn"; remove.textContent = course.deleting ? "重试删除" : "删除"; remove.addEventListener("click", () => deleteCourse(id, name, remove, row));
        if (course.deleting) { const note = document.createElement("p"); note.textContent = "删除尚未完成，课程已暂停使用，请重试删除。"; info.append(note); }
        actions.append(remove); row.append(info, actions); container.append(row);
      }
    } catch (error) { container.textContent = "课程加载失败：" + error.message; }
  }
  async function deleteCourse(id, name, button, row) {
    if (button.disabled || !confirm("确定彻底删除课程“" + name + "”？该课程的全部节点、关系、上传课件和检索资料会一并删除，无法撤销。")) return;
    const status = document.getElementById("courseStatus");
    button.disabled = true; button.textContent = "正在删除…"; row.setAttribute("aria-busy", "true");
    status.textContent = "正在删除课程“" + name + "”及相关资料，请稍候…";
    try {
      await A.request("/api/courses/" + encodeURIComponent(id), {method: "DELETE"});
      status.textContent = "课程“" + name + "”及相关资料已删除。"; await loadCourses();
    } catch (error) {
      status.textContent = "删除未完成：" + error.message;
      button.textContent = "重试删除";
    } finally { button.disabled = false; row.removeAttribute("aria-busy"); }
  }
  window.addCourse = async () => {
    const name = prompt("请输入课程名称："); if (!name || !name.trim()) return;
    try { await A.request("/api/courses", {method: "POST", json: {name: name.trim()}}); await loadCourses(); }
    catch (error) { alert("创建失败：" + error.message); }
  };
  window.addEventListener("DOMContentLoaded", loadCourses);
})();
