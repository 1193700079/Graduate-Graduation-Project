
// document.getElementById("login-form").addEventListener("submit", function (e) {
//   e.preventDefault();
//   const username = document.getElementById("login-username").value;
//   const password = document.getElementById("login-password").value;
//   const storedUsername = localStorage.getItem("reg-username");
//   const storedPassword = localStorage.getItem("reg-password");
//     const rememberMe = document.getElementById("remember-me").checked;
//     console.log(username);
//     console.log(password);
//     console.log(storedUsername);
//     console.log(storedPassword);
//   if (username === storedUsername && password === storedPassword) {
//     if (rememberMe) {
//       localStorage.setItem("rememberMe", "true");
//       localStorage.setItem("rememberedUsername", username);
//       localStorage.setItem("rememberedPassword", password);
//     } else {
//       localStorage.removeItem("rememberMe");
//       localStorage.removeItem("rememberedUsername");
//       localStorage.removeItem("rememberedPassword");
//     }
//     window.location.href = "http://121.43.138.58:7860"; // 登录成功后的跳转
//   } else {
//     alert("Invalid username or password.");
//   }
// });

// 当页面加载时，检查是否记住了用户信息
window.onload = function () {
  if (localStorage.getItem("rememberMe") === "true") {
    document.getElementById("login-username").value =
      localStorage.getItem("rememberedUsername");
    document.getElementById("login-password").value =
      localStorage.getItem("rememberedPassword");
    document.getElementById("remember-me").checked = true;
  }
};
