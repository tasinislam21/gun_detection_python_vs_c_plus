// Check if getUserMedia is supported and load frame into video

var socket = io.connect('http://' + document.domain + ':' + location.port);
socket.on('connect', function() {
    console.log("Socket connected");
    requestFrame();
});

function requestFrame(){
    socket.emit('image');
    setTimeout(requestFrame, 33);
}

var img = document.createElement('img');
img.classList.add('img-fluid', 'border', 'rounded', 'mt-3');

const container = document.getElementById('imageContainer');
container.appendChild(img);

socket.on('response_back', function(data) {
    img.src = data;
});