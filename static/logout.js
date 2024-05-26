window.onbeforeunload = function() {
    fetch('/logout', { method: 'POST', credentials: 'same-origin' });
}
