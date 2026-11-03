(function () {
  "use strict";
  const A = window.CourseApp, send = document.getElementById("sendBtn"), input = document.getElementById("input"), messages = document.getElementById("messages");
  const selector = document.getElementById("chatCourse"), notice = document.getElementById("chatCourseNotice"); let courses = [], busy = false;
  function message(value, sender) {
    const element = document.createElement("div"); element.className = "message " + sender; element.style.whiteSpace = "pre-wrap"; element.textContent = value; messages.append(element); messages.scrollTop = messages.scrollHeight;
    if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise([element]).catch(() => {}); return element;
  }
  async function loadCourses() {
    try {
      const data = await A.request("/api/courses"); courses = Array.isArray(data) ? data : data.courses || [];
      const selected = new URLSearchParams(location.search).get("course_id") || ""; selector.replaceChildren();
      const placeholder = document.createElement("option"); placeholder.value = ""; placeholder.textContent = "请选择提问课程"; selector.append(placeholder);
      for (const course of courses) { const option = document.createElement("option"); option.value = course.course_id || course.id; option.textContent = course.course_name || course.name; selector.append(option); }
      if (courses.some(course => String(course.course_id || course.id) === selected)) selector.value = selected;
      notice.textContent = courses.length ? "优先结合课程资料回答并提供引用；资料不足时使用通用知识，并明确标注。" : "尚无课程，请先由教师创建课程；没有课件也可进行通用知识问答。";
    } catch (error) { notice.textContent = "课程列表加载失败：" + error.message; }
  }
  async function ask() {
    if (busy || !input.value.trim()) return;
    const course = courses.find(item => String(item.course_id || item.id) === selector.value);
    if (!course) { notice.textContent = "请先选择课程，再发送问题。"; selector.focus(); return; }
    const question = input.value.trim(); input.value = ""; message(question, "user"); const answer = message("正在查找课程资料并生成回答……", "ai"); busy = true; send.disabled = true; messages.setAttribute("aria-busy", "true");
    try {
      const data = await A.request("/api/qa/", {method: "POST", json: {question, course_id: course.course_id || course.id, course_name: course.course_name || course.name}});
      if (!data.answer) throw Error("未返回答案，请稍后重试"); answer.textContent = data.answer;
      if (data.answer_mode !== "general" && data.retrieval === "course_keyword_graph") {
        const note = document.createElement("p"); note.className = "retrieval-note"; note.textContent = "本次使用课程关键词与图谱资料检索，未启用向量检索。"; answer.append(note);
      }
      if (Array.isArray(data.sources) && data.sources.length) {
        const title = document.createElement("strong"); title.textContent = "参考来源";
        const list = document.createElement("ul"); list.className = "answer-sources";
        for (const source of data.sources) {
          const item = document.createElement("li");
          const origin = source.source_type === "teacher_edit" ? "教师编辑：" : "";
          const review = source.source_type === "teacher_edit" && source.review_status === "pending" ? "（待确认）" : "";
          const label = (source.citation_id ? "[" + source.citation_id + "] " : "") + origin + (source.name || "课程资料") + review + (source.page_no ? " · 第 " + source.page_no + " 页" : "");
          const target = A.localSourceUrl(source.source_url);
          if (target) {
            const link = document.createElement("a"); link.textContent = source.source_type === "teacher_edit" ? "查看原课件" : label; link.href = target; link.target = "_blank"; link.rel = "noopener";
            if (source.source_type === "teacher_edit") item.append(document.createTextNode(label + " · "));
            item.append(link);
          }
          else item.textContent = label;
          list.append(item);
        }
        answer.append(title, list);
      }
      if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise([answer]).catch(() => {});
    } catch (error) { answer.textContent = "本次回答未完成：" + error.message; answer.classList.add("error"); }
    finally { busy = false; send.disabled = false; messages.setAttribute("aria-busy", "false"); }
  }
  window.addEventListener("DOMContentLoaded", () => {
    loadCourses(); send.addEventListener("click", ask); input.addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(); } });
  });
})();
