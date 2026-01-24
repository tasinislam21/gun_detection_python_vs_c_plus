var video = document.getElementById('video');
video.style.display = "none";

// Check if getUserMedia is supported and load frame into video
video.src = "../evaluation.mp4";
video.play();

var socket = io.connect('http://' + document.domain + ':' + location.port);
socket.on('connect', function() {
    console.log("Socket connected");
});

video.addEventListener('play', function() {
    var canvas = document.createElement('canvas');
    canvas.width = video.width;
    canvas.height = video.height;
    var context = canvas.getContext('2d');

    function sendFrame() {
        if (video.paused || video.ended) {
            console.log("Video paused or ended");
            return;
        }

        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        var dataURL = canvas.toDataURL('image/jpeg');
        console.log("Sending frame...");
        socket.emit('image', dataURL);

        requestAnimationFrame(sendFrame);
    }

    console.log("Starting to send frames...");
    requestAnimationFrame(sendFrame);
}, false);
var img = document.createElement('img');

socket.on('response_back', function(data) {
    img.src = data;
    img.classList.add('img-fluid', 'border', 'rounded', 'mt-3');
    document.getElementById('imageContainer').appendChild(img);
    console.log("Received processed frame");
});