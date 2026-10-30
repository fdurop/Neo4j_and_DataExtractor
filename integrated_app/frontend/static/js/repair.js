(function () {
  "use strict";
  const A = window.CourseApp, query = new URLSearchParams(location.search);
  const course = {courseId: query.get("course_id") || "", courseName: query.get("course_name") || ""};
  function path(url) { return url + (course.courseId ? (url.includes("?") ? "&" : "?") + "course_id=" + encodeURIComponent(course.courseId) : ""); }
  function el(tag, value, cls) { const node = document.createElement(tag); if (value !== undefined) node.textContent = value; if (cls) node.className = cls; return node; }
  async function load() {
    const list = document.getElementById("errorList"), context = document.getElementById("reviewCourse");
    context.textContent = course.courseId ? "当前课程：" + (course.courseName || course.courseId) : "查看全部课程素材；每条记录会标明所属课程。";
    try {
      const data = await A.request(path("/api/repair/items"));
      const items = (Array.isArray(data.items) ? data.items : []).filter(item => !course.courseId || item.course_id === course.courseId);
      list.replaceChildren(); document.getElementById("itemCount").textContent = "（" + items.length + " 条）";
      if (!items.length) { list.append(el("p", "当前没有可核对的素材。请先上传并完成课件处理。", "loading-text")); return; }
      for (const item of items) {
        const card = el("article", undefined, "error-card"), preview = el("div", undefined, "image-preview");
        const imageUrl = A.localImageUrl(item.image_url), sourceUrl = A.localSourceUrl(item.source_url);
        if (imageUrl) {
          const image = document.createElement("img"); image.alt = item.name ? item.name + "的课件图片" : "关联课件图片"; image.src = imageUrl;
          image.addEventListener("error", () => preview.replaceChildren(el("p", "局部图片暂不可用，请打开原课件核对。", "preview-note")), {once: true}); preview.append(image);
        } else preview.append(el("p", "此素材没有可用的局部图片，可打开原课件核对。", "preview-note"));
        const edit = el("div", undefined, "edit-section");
        const badge = el("span", ["approved", "reviewed"].includes(item.review_status) ? "文字已核对" : "等待人工核对", "status-badge");
        edit.append(badge, el("p", "所属课程：" + (item.course_name || item.course_id || "待确认") + (item.page_no ? " · 第 " + item.page_no + " 页" : ""), "course-context"));
        edit.append(el("h3", item.name || item.type || "素材描述"));
        if (sourceUrl) {
          const source = el("a", item.page_no ? "打开原课件第" + item.page_no + "页" : "打开原课件", "source-link");
          source.href = sourceUrl; source.target = "_blank"; source.rel = "noopener"; edit.append(source);
        }
        const label = el("label", "素材文字描述"), input = document.createElement("textarea"); input.value = item.description || ""; input.setAttribute("aria-label", "素材文字描述"); label.append(input); edit.append(label);
        const button = el("button", "保存文字修订", "save-btn"), result = el("p", "", "save-result"); result.setAttribute("role", "status");
        button.addEventListener("click", async () => {
          if (!input.value.trim()) { result.textContent = "描述内容不能为空"; result.classList.add("error"); return; }
          if (!item.course_id) { result.textContent = "这条素材缺少课程标识，未提交修改"; result.classList.add("error"); return; }
          button.disabled = true; result.textContent = "正在保存……"; result.classList.remove("error");
          try {
            const saved = await A.request("/api/repair/items/" + encodeURIComponent(item.id), {method: "PUT", json: {course_id: item.course_id, description: input.value.trim()}});
            result.textContent = saved.message || "素材文字修订已保存，相关概念和关系仍需核对。";
            badge.textContent = "文字修订已保存"; button.textContent = "再次保存文字修订";
          } catch (error) { result.textContent = "保存失败：" + error.message; result.classList.add("error"); }
          finally { button.disabled = false; }
        });
        edit.append(button, result); card.append(preview, edit); list.append(card);
      }
    } catch (error) { list.textContent = "获取待核对素材失败：" + error.message; }
  }
  document.addEventListener("DOMContentLoaded", load);
})();
