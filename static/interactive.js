const videoFeed = document.getElementById('video-feed');
const videoCanvas = document.getElementById('video-canvas');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const hazardCount = document.getElementById('hazard-count');
const sensitivitySlider = document.getElementById('sensitivity');
const sensVal = document.getElementById('sens-val');
const snapshotBtn = document.getElementById('snapshot-btn');

let lastHazard = false;
let lastCount = 0;

// --- Sensitivity slider ---
sensitivitySlider.addEventListener('input', function() {
    sensVal.textContent = this.value;
    fetch('/set_sensitivity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sensitivity: this.value })
    });
});

// --- Poll status and update UI ---
function updateStatus() {
    fetch('/machine_status')
        .then(response => response.json())
        .then(data => {
            if (data.auto_stop) {
                statusDot.classList.remove('safe');
                statusDot.classList.add('hazard');
                statusText.textContent = 'STOPPED (Hazard Detected)';
                if (!lastHazard) showNotification('Hazard Detected! Machine Stopped.', 'danger');
            } else {
                statusDot.classList.remove('hazard');
                statusDot.classList.add('safe');
                statusText.textContent = 'Safe';
            }
            hazardCount.textContent = data.hazard_count;
            lastHazard = data.auto_stop;
            lastCount = data.hazard_count;
        });
}
setInterval(updateStatus, 2000);

// --- Draw overlays on video ---
function drawOverlay() {
    if (!videoFeed.complete) return requestAnimationFrame(drawOverlay);
    const w = videoFeed.naturalWidth;
    const h = videoFeed.naturalHeight;
    if (!w || !h) return requestAnimationFrame(drawOverlay);
    videoCanvas.width = w;
    videoCanvas.height = h;
    const ctx = videoCanvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    // Draw animated rectangle (machine highlight)
    const rw = w * 0.4, rh = h * 0.4;
    const x1 = w/2 - rw/2, y1 = h/2 - rh/2, x2 = x1 + rw, y2 = y1 + rh;
    const now = Date.now()/500;
    let color = statusDot.classList.contains('hazard') ? `rgba(185,28,28,${0.7+0.3*Math.abs(Math.sin(now))})` : `rgba(26,127,55,0.7)`;
    ctx.lineWidth = 6;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, rw, rh);
    // Label
    ctx.font = 'bold 32px Arial';
    ctx.textAlign = 'center';
    ctx.fillStyle = color;
    ctx.fillText('Machine', w/2, y1-15 < 30 ? y1+40 : y1-15);
    requestAnimationFrame(drawOverlay);
}
videoFeed.onload = drawOverlay;
videoFeed.onresize = drawOverlay;
drawOverlay();

// --- Snapshot download ---
snapshotBtn.addEventListener('click', function() {
    fetch('/snapshot')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'snapshot.jpg';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            showNotification('Snapshot downloaded!', 'success');
        });
});

// --- Toast/notification system ---
function showNotification(message, type) {
    const area = document.getElementById('notification-area');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} fade show`;
    alert.textContent = message;
    area.appendChild(alert);
    setTimeout(() => {
        alert.classList.remove('show');
        alert.classList.add('hide');
        setTimeout(() => area.removeChild(alert), 500);
    }, 2000);
} 