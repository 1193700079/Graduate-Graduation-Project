document.getElementById('register-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const username = document.getElementById('reg-username').value;
    const password = document.getElementById('reg-password').value;
    localStorage.setItem("reg-username", username);
    localStorage.setItem("reg-password", password);
    alert('Registration successful!');
    // window.location.href = '/'; // Redirect to login page after registration
});
