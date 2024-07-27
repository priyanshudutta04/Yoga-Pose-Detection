import cv2
from ultralytics import YOLO
import cvzone
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


yoga_poses = [
    "adho mukha svanasana",
    "adho mukha vriksasana",
    "agnistambhasana",
    "ananda balasana",
    "anantasana",
    "anjaneyasana",
    "ardha bhekasana",
    "ardha chandrasana",
    "ardha matsyendrasana",
    "ardha pincha mayurasana",
    "ardha uttanasana",
    "ashtanga namaskara",
    "astavakrasana",
    "baddha konasana",
    "bakasana",
    "balasana",
    "bhairavasana",
    "bharadvajasana i",
    "bhekasana",
    "bhujangasana",
    "bhujapidasana",
    "bitilasana",
    "camatkarasana",
    "chakravakasana",
    "chaturanga dandasana",
    "dandasana",
    "dhanurasana",
    "durvasasana",
    "dwi pada viparita dandasana",
    "eka pada koundinyanasana i",
    "eka pada koundinyanasana ii",
    "eka pada rajakapotasana",
    "eka pada rajakapotasana ii",
    "ganda bherundasana",
    "garbha pindasana",
    "garudasana",
    "gomukhasana",
    "halasana",
    "hanumanasana",
    "janu sirsasana",
    "kapotasana",
    "krounchasana",
    "kurmasana",
    "lolasana",
    "makara adho mukha svanasana",
    "makarasana",
    "malasana",
    "marichyasana i",
    "marichyasana iii",
    "marjaryasana",
    "matsyasana",
    "mayurasana",
    "natarajasana",
    "padangusthasana",
    "padmasana",
    "parighasana",
    "paripurna navasana",
    "parivrtta janu sirsasana",
    "parivrtta parsvakonasana",
    "parivrtta trikonasana",
    "parsva bakasana",
    "parsvottanasana",
    "pasasana",
    "paschimottanasana",
    "phalakasana",
    "pincha mayurasana",
    "prasarita padottanasana",
    "purvottanasana",
    "salabhasana",
    "salamba bhujangasana",
    "salamba sarvangasana",
    "salamba sirsasana",
    "savasana",
    "setu bandha sarvangasana",
    "simhasana",
    "sukhasana",
    "supta baddha konasana",
    "supta matsyendrasana",
    "supta padangusthasana",
    "supta virasana",
    "tadasana",
    "tittibhasana",
    "tolasana",
    "tulasana",
    "upavistha konasana",
    "urdhva dhanurasana",
    "urdhva hastasana",
    "urdhva mukha svanasana",
    "urdhva prasarita eka padasana",
    "ustrasana",
    "utkatasana",
    "uttana shishosana",
    "uttanasana",
    "utthita ashwa sanchalanasana",
    "utthita hasta padangustasana",
    "utthita parsvakonasana",
    "utthita trikonasana",
    "vajrasana",
    "vasisthasana",
    "viparita karani",
    "virabhadrasana i",
    "virabhadrasana ii",
    "virabhadrasana iii",
    "virasana",
    "vriksasana",
    "vrischikasana",
    "yoganidrasana"
]

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", 
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "mobile phone","microwave","oven","toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"
]

def imgProcess(image):
    target_size=(64, 64)
    img = cv2.resize(image, target_size)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 
    output=predict(img)
    return output

def predict(test_inp):
    yoga_model = tf.keras.Sequential([
        
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
        
        
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
        
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
        
        tf.keras.layers.Flatten(),
        
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(107, activation='softmax')
    ])
    yoga_model.load_weights('yoga-model.h5')

    prediction=yoga_model.predict(test_inp)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index


def yogaPoseDetect():
    model=YOLO('yolov8n.pt')
    

    cap=cv2.VideoCapture(0)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    # fps = cap.get(cv2.CAP_PROP_FPS) 
    # out = cv2.VideoWriter('sample/demo.avi', fourcc, fps, (1280, 720))   
    cap.set(3,1280)
    cap.set(4,720)

    while True:
        success, img=cap.read()
        results=model(img,stream=True)

        if not success:
            break

        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]                                
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                conf=round(float(box.conf[0]),2)                        
                id=int(box.cls[0])                                     
                class_name = classNames[id]

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)         


                if class_name == "person":
                    cropped_img = img[y1:y2, x1:x2]
                    predicted_pose = imgProcess(cropped_img)
                    cvzone.putTextRect(img,f'{yoga_poses[predicted_pose]}',(max(0,x1),max(40,y1)))
                       
        # out.write(img)

        cv2.imshow("Cam footage. Press 'Q' to exit.",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()



yogaPoseDetect()
 


