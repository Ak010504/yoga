// Global state
const state = {
    selectedPose: null,
    backendURL: 'https://nonculminating-conductible-francina.ngrok-free.dev',
    recordedBlob: null,
    mediaRecorder: null,
    recordingStartTime: null,
    recordingInterval: null,
    videoURL: null   // 🔹 add this
};

const NGROK_HEADERS = {
    'ngrok-skip-browser-warning': 'true'
};



// Initialize app on load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('PosePilot PWA Loading...');
    
    // Load poses from backend
    await loadPoses();
    
    // Setup file upload handler
    const videoFile = document.getElementById('video-file');
    videoFile.addEventListener('change', handleFileSelect);
    
    // Setup drag & drop
    const uploadArea = document.getElementById('upload-area');
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = '';
        if (e.dataTransfer.files.length > 0) {
            videoFile.files = e.dataTransfer.files;
            handleFileSelect({target: {files: e.dataTransfer.files}});
        }
    });
    
    // Register service worker for PWA
    if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/service-worker.js')
      .then(reg => {
        console.log('✅ Service Worker registered with scope:', reg.scope);
      })
      .catch(err => {
        console.error('❌ Service Worker registration failed:', err);
      });
  });
}

});

// Load available poses from backend
// Load available poses from backend
async function loadPoses() {
    try {
        const url = `${state.backendURL}/api/poses`;
        console.log('🔎 Fetching poses from:', url);

        const res = await fetch(url, { headers: NGROK_HEADERS });

        const contentType = res.headers.get('content-type') || '';
        const text = await res.text();

        console.log('📥 Raw /api/poses response status:', res.status);
        console.log('📥 Raw /api/poses content-type:', contentType);
        console.log('📥 Raw /api/poses body (first 300 chars):', text.slice(0, 300));

        // If it's not JSON, don't try to parse it
        if (!contentType.includes('application/json')) {
            console.error('❌ Backend did not return JSON. See raw response above.');
            alert('Backend returned non-JSON for /api/poses (see console log).');
            return;
        }

        const data = JSON.parse(text);  // now safe

        const dropdown = document.getElementById('pose-dropdown');
        dropdown.innerHTML = '<option value="">-- Choose a pose --</option>';

        data.poses.forEach(pose => {
            const option = document.createElement('option');
            option.value = pose;
            option.textContent = pose.charAt(0).toUpperCase() + pose.slice(1).replace(/_/g, ' ');
            dropdown.appendChild(option);
        });

        dropdown.addEventListener('change', (e) => {
            state.selectedPose = e.target.value;
        });

        console.log('✅ Poses loaded:', data.poses);
    } catch (error) {
        console.error('❌ loadPoses network error:', error);
        alert('Could not reach backend (network error). Check console.');
    }
}



// Screen navigation
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

function backHome() {
    state.recordedBlob = null;
    showScreen('home-screen');
}

function showUploadScreen() {
    if (!state.selectedPose) {
        alert('Please select a pose first');
        return;
    }
    showScreen('upload-screen');
    document.getElementById('file-info').style.display = 'none';
    document.getElementById('upload-progress').style.display = 'none';
}

function showRecordScreen() {
    if (!state.selectedPose) {
        alert('Please select a pose first');
        return;
    }
    showScreen('record-screen');
    initializeRecording();
}

// Upload video handling
function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length === 0) return;
    
    const file = files[0];
    if (!file.type.startsWith('video/')) {
        alert('Please select a video file');
        return;
    }
    
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-info').style.display = 'block';
}

async function uploadVideo() {
    const videoFile = document.getElementById('video-file').files[0];
    if (!videoFile) {
        alert('Please select a video file');
        return;
    }
    
    await analyzeVideo(videoFile);
}

// Record video handling
async function initializeRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });
        
        const video = document.getElementById('record-video');
        video.srcObject = stream;
        
        document.getElementById('recording-view').style.display = 'block';
        document.getElementById('recorded-view').style.display = 'none';
        document.getElementById('permission-denied').style.display = 'none';
        
    } catch (error) {
        console.error('Camera access denied:', error);
        document.getElementById('recording-view').style.display = 'none';
        document.getElementById('permission-denied').style.display = 'block';
    }
}

function startRecording() {
    const video = document.getElementById('record-video');
    const stream = video.srcObject;
    
    if (!stream) {
        alert('Camera not available');
        return;
    }
    
    state.mediaRecorder = new MediaRecorder(stream);
    const chunks = [];
    
    state.mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    state.mediaRecorder.onstop = () => {
        state.recordedBlob = new Blob(chunks, { type: 'video/webm' });
        showRecordedVideo();
    };
    
    state.mediaRecorder.start();
    state.recordingStartTime = Date.now();
    
    document.getElementById('start-record-btn').style.display = 'none';
    document.getElementById('stop-record-btn').style.display = 'block';
    
    startRecordingTimer();
}

function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
        clearInterval(state.recordingInterval);
        document.getElementById('recording-timer').textContent = '00:00';
        
        // Stop all tracks
        const video = document.getElementById('record-video');
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
    }
    
    document.getElementById('start-record-btn').style.display = 'block';
    document.getElementById('stop-record-btn').style.display = 'none';
}

function startRecordingTimer() {
    state.recordingInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.recordingStartTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        document.getElementById('recording-timer').textContent = 
            `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }, 100);
}

function showRecordedVideo() {
    document.getElementById('recording-view').style.display = 'none';
    document.getElementById('recorded-view').style.display = 'block';
    
    const videoURL = URL.createObjectURL(state.recordedBlob);
    document.getElementById('playback-video').src = videoURL;
}

function discardRecording() {
    state.recordedBlob = null;
    document.getElementById('recorded-view').style.display = 'none';
    document.getElementById('recording-view').style.display = 'block';
    
    // Restart camera
    initializeRecording();
}

async function analyzeRecordedVideo() {
    if (!state.recordedBlob) {
        alert('No video recorded');
        return;
    }
    
    const file = new File([state.recordedBlob], 'recorded-video.webm', { type: 'video/webm' });
    await analyzeVideo(file);
}

// Video analysis (upload or recorded)
async function analyzeVideo(videoFile) {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('pose', state.selectedPose);

    // 🔹 Create / update video URL for results screen
    if (state.videoURL) {
        URL.revokeObjectURL(state.videoURL);
    }
    state.videoURL = URL.createObjectURL(videoFile);

    showScreen('loading-screen');
    document.getElementById('loading-text').textContent =
        'Processing your video... This may take 30-60 seconds.';

    try {
        const response = await fetch(`${state.backendURL}/api/process-video`, {
            method: 'POST',
            body: formData,
            headers: NGROK_HEADERS
        });

        const result = await response.json();

        if (response.ok) {
            displayResults(result);
        } else {
            alert(`Error: ${result.error || 'Failed to process video'}`);
            backHome();
        }
    } catch (error) {
        console.error('Analysis error:', error);
        alert(`Failed to analyze video: ${error.message}`);
        backHome();
    }
}


// Display results
function displayResults(result) {
    showScreen('results-screen');

    document.getElementById('result-pose').textContent =
        state.selectedPose.charAt(0).toUpperCase() +
        state.selectedPose.slice(1).replace(/_/g, ' ');

    document.getElementById('result-frames').textContent =
        result.feedback_log?.length || 10;

    // 🔹 Set video source if available
    const resultVideo = document.getElementById('result-video');
    if (resultVideo && state.videoURL) {
        resultVideo.src = state.videoURL;
        resultVideo.load();
    }

    const feedbackList = document.getElementById('feedback-list');
    feedbackList.innerHTML = '';

    if (result.feedback_log && result.feedback_log.length > 0) {
        result.feedback_log.forEach((fb, index) => {
            const item = document.createElement('div');
            item.className = `card feedback-item ${fb.severity || 'positive'}`;

            const timestamp = fb.timestamp
                ? `${fb.timestamp.toFixed(1)}s`
                : `Frame ${index + 1}`;

            item.innerHTML = `
                <div class="feedback-header">
                    <span>Frame ${index + 1} / 10</span>
                    <span class="feedback-timestamp">${timestamp}</span>
                </div>
                <div class="feedback-text">${fb.feedback || 'Analysis complete'}</div>
                <div class="feedback-actions">
                    <button class="btn btn-primary" style="font-size: 12px; padding: 6px 12px;"
                        onclick="speakFeedback('${fb.feedback}')">
                        🔊 Hear Feedback
                    </button>
                </div>
            `;

            feedbackList.appendChild(item);
        });
    } else {
        feedbackList.innerHTML =
            '<div class="card"><p>Analysis complete. No feedback available.</p></div>';
    }
}


// Text-to-speech feedback
function speakFeedback(text) {
    if (!text) return;
    
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    synth.speak(utterance);
}

// Download results as JSON
function downloadResults() {
    const pose = state.selectedPose;
    const results = {
        pose: pose,
        timestamp: new Date().toISOString(),
        feedback: Array.from(document.querySelectorAll('.feedback-item')).map((item, index) => ({
            frame: index + 1,
            feedback: item.querySelector('.feedback-text')?.textContent || ''
        }))
    };
    
    const jsonString = JSON.stringify(results, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `posepilot-${pose}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
