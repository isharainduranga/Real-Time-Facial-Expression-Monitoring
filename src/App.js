import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

function App() {
 
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [faces, setFaces] = useState([]);

  // Start video stream from the webcam
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
      })
      .catch((err) => {
        console.error("Error accessing webcam: ", err);
      });
  }, []);

  // Capture a frame and send it to the backend
  const captureFrame = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas image to base64
    const image = canvas.toDataURL('image/jpeg');

    // Send the image to the backend for prediction
    axios.post('http://127.0.0.1:5000/predict_emotion', { image })
      .then((response) => {
        setFaces(response.data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  // Capture a frame every 1 second
  useEffect(() => {
    const interval = setInterval(captureFrame, 1000);
    return () => clearInterval(interval);
  }, []);


  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-r from-blue-500 to-indigo-600">
    <h1 className="text-4xl font-bold text-white mb-8">Real-time Emotion Detection</h1>

    <div className="relative rounded-lg overflow-hidden border-4 border-white shadow-lg">
      <video
        ref={videoRef}
        autoPlay
        className="w-full h-full object-cover"
      ></video>
      <canvas ref={canvasRef} className="hidden"></canvas>

      {/* Render detected faces */}
      {faces.map((face, index) => (
        <div
          key={index}
          className="absolute border-4 border-red-500"
          style={{
            left: `${face.coordinates[0]}px`,
            top: `${face.coordinates[1]}px`,
            width: `${face.coordinates[2]}px`,
            height: `${face.coordinates[3]}px`,
          }}
        >
          <span className="bg-red-500 text-white text-sm px-2 rounded absolute top-0 left-0">
            {face.emotion}
          </span>
        </div>
      ))}
    </div>

    <div className="mt-8 p-4 bg-white rounded-lg shadow-lg text-center">
      <h2 className="text-2xl font-semibold text-gray-700">Detected Faces:</h2>
      {faces.length > 0 ? (
        <ul className="list-disc">
          {faces.map((face, index) => (
            <li key={index} className="text-lg text-indigo-600">
              Face {index + 1}: {face.emotion}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-lg text-gray-600">No faces detected</p>
      )}
    </div>
  </div>
  );
}

export default App;
