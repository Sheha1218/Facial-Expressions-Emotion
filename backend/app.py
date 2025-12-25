from fastapi import FastAPI,Response,Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import torch
import cv2
import numpy as np
from backend.detector import classifire



CLASS_NAMES = ['angry','confused','disgust','fear','happy','neutral','sad','shy','suprise']

app = FastAPI()
frontend = Jinja2Templates(directory='frontend')


classifire = classifire(r'D:\Way to Denmark\Projects\Facial-Expressions-Emotion\model\model.pth',CLASS_NAMES)

cap =cv2.VideoCapture(0)

def preprocess(frame):
    frame_resized = cv2.resize(frame,(128,128))
    frame_rgb =cv2.cvtColor(frame_resized)
    frame_rgb =frame_rgb.astype(np.float32)/255.0
    frame_transposed =np.transpose(frame_rgb,(2,0,1))
    frame_exp=np.expand_dims(frame_transposed,axis=0)
    return torch.tensor(frame_exp)

def generate_frame():
    while True:
        ret,frame = cap.read()
        if not ret:
            continue
        
        input_tensor = preprocess(frame)
        label,confidence = classifire.predict(input_tensor)
        
        cv2.rectangle(frame,(0,0),frame.shape[1],60,(30,30,30),-1)
        cv2.putText(frame,f"class:{label} | Conf:{int(confidence*100)}%",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)
        
        ret,buffer =cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
        
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return frontend.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/video")
def video():
    return Response(
        generate_frame(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
    
if __name__ == '__main__':
    app.run(host='5000',debug=True)
        
        

