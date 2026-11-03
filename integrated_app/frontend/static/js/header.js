window.addEventListener("DOMContentLoaded", async () => {
  "use strict";
  const roleButton = document.getElementById("roleBtn"), loginButton = document.getElementById("loginBtn");
  let session = {authenticated: false};
  try { session = await window.CourseApp.request("/api/session"); } catch (_) {}
  const user = session.user || session;
  if (roleButton) {
    roleButton.textContent = session.authenticated ? "账号：" + user.username + "（" + (user.role === "teacher" ? "教师" : "学生") + "）" : "未登录";
    roleButton.title = "打开账号对应的课程页面";
    roleButton.addEventListener("click", () => { location.href = session.authenticated ? user.role === "teacher" ? "/teacher/" : "/" : "/login/"; });
  }
  if (loginButton) {
    loginButton.textContent = session.authenticated ? "退出登录" : "登录";
    loginButton.addEventListener("click", async () => {
      if (session.authenticated) { try { await window.CourseApp.request("/logout", {method: "POST"}); } catch (error) { alert(error.message); return; } }
      localStorage.removeItem("qa_auth_token"); localStorage.removeItem("qa_user_role"); location.href = "/login/";
    });
  }
  document.getElementById("clearBtn")?.addEventListener("click", () => { document.getElementById("messages")?.replaceChildren(); });
});
window.headerLoader = {load: async () => {}, waitForLoad: async () => {}};
