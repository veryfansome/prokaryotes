(function () {
    const statusMessage = document.getElementById('statusMessage');
    if (!statusMessage) {
        return;
    }

    const params = new URLSearchParams(window.location.search);
    const error = params.get('error');
    const info = params.get('info');

    if (error) {
        statusMessage.textContent = error;
        statusMessage.classList.add('error');
        statusMessage.hidden = false;
    } else if (info) {
        statusMessage.textContent = info;
        statusMessage.classList.add('info');
        statusMessage.hidden = false;
    }
})();
