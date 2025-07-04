// --- Live hazard sensitivity adjustment ---
const sensitivitySlider = document.getElementById('sensitivity');
const sensVal = document.getElementById('sens-val');
sensitivitySlider.addEventListener('input', function() {
    sensVal.textContent = this.value;
    fetch('/set_sensitivity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sensitivity: this.value })
    });
});

// --- Input source switching (image upload placeholder) ---
const inputSource = document.getElementById('input-source');
const imageUploadSection = document.getElementById('image-upload-section');
inputSource.addEventListener('change', function() {
    if (this.value === 'image') {
        imageUploadSection.style.display = '';
        document.getElementById('video-feed').style.display = 'none';
        document.getElementById('snapshot-btn').style.display = 'none';
    } else {
        imageUploadSection.style.display = 'none';
        document.getElementById('video-feed').style.display = '';
        document.getElementById('snapshot-btn').style.display = '';
    }
});

// --- Live status, hazard count, and animated status dot ---
function updateStatus() {
    fetch('/machine_status')
        .then(response => response.json())
        .then(data => {
            const statusDiv = document.getElementById('machine-status');
            const statusText = document.getElementById('status-text');
            const statusIcon = document.getElementById('status-icon');
            const hazardCount = document.getElementById('hazard-count');
            if (data.auto_stop) {
                statusDiv.classList.remove('alert-success');
                statusDiv.classList.add('alert-danger');
                statusText.textContent = 'STOPPED (Hazard Detected)';
                statusIcon.classList.remove('bg-success');
                statusIcon.classList.add('bg-danger');
                showNotification('Hazard Detected! Machine Stopped.', 'danger');
            } else {
                statusDiv.classList.remove('alert-danger');
                statusDiv.classList.add('alert-success');
                statusText.textContent = 'Safe';
                statusIcon.classList.remove('bg-danger');
                statusIcon.classList.add('bg-success');
            }
            hazardCount.textContent = data.hazard_count;
        });
}
setInterval(updateStatus, 2000);

// --- Snapshot download ---
document.getElementById('snapshot-btn').addEventListener('click', function() {
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