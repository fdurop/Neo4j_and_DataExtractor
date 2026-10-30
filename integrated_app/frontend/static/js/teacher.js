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
        const remove = document.createElement("button"); remove.className = "delete-btn"; remove.textContent = "删除"; remove.addEventListener("click", () => deleteCourse(id));
        actions.append(remove); row.append(info, actions); container.append(row);
      }
    } catch (error) { container.textContent = "课程加载失败：" + error.message; }
  }
  async function deleteCourse(id) {
    if (!confirm("确定删除该课程？")) return;
    try { await A.request("/api/courses/" + encodeURIComponent(id), {method: "DELETE"}); await loadCourses(); }
    catch (error) { alert("删除失败：" + error.message); }
  }
  window.addCourse = async () => {
    const name = prompt("请输入课程名称："); if (!name || !name.trim()) return;
    try { await A.request("/api/courses", {method: "POST", json: {name: name.trim()}}); await loadCourses(); }
    catch (error) { alert("创建失败：" + error.message); }
  };
  window.addEventListener("DOMContentLoaded", loadCourses);
})();
