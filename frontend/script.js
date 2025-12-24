const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const labelText = document.getElementById("label");
const confText = document.getElementById("conf");
const fillBar = document.getElementById("fill");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => video.srcObject = stream);

canvas.width = 160;
canvas.height = 160;

setInterval(() => {
    ctx.drawImage(video, 0, 0, 160, 160);
    const image = canvas.toDataURL("image/jpeg");

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image })
    })
    .then(res => res.json())
    .then(data => {
        labelText.innerText = `Class: ${data.label}`;
        confText.innerText = `Confidence: ${data.confidence * 100}%`;
        fillBar.style.width = `${data.confidence * 100}%`;
    });
}, 300);  // adjust speed
