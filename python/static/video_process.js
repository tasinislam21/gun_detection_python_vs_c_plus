var video = document.getElementById('video');
video.style.display = "none";

// Check if getUserMedia is supported and load frame into video

var socket = io.connect('http://' + document.domain + ':' + location.port);
socket.on('connect', function() {
    console.log("Socket connected");
});

var canvas = document.createElement('canvas');
canvas.width = video.width;
canvas.height = video.height;
var context = canvas.getContext('2d');

function sendFrame() {
    if (!video.videoWidth || !video.videoHeight) {
        requestAnimationFrame(sendFrame);
        return;
    }

    if (video.paused) {
        video.play();
        requestAnimationFrame(sendFrame);
        return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    var dataURL = canvas.toDataURL('image/jpeg', 0.7);
    socket.emit('image', dataURL);
    requestAnimationFrame(sendFrame);
}


console.log("Starting to send frames...");
requestAnimationFrame(sendFrame);

var img = document.createElement('img');

socket.on('response_back', function(data) {
    img.src = data;
    img.classList.add('img-fluid', 'border', 'rounded', 'mt-3');

    const container = document.getElementById('imageContainer');

    if (!container.firstChild) {
        container.appendChild(img);
    } else {
        container.firstChild.src = data;
    }

    console.log("Received processed frame");
});