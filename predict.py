import cv2
from fastai.vision.all import load_learner, PILImage
import numpy as np

# Load the saved model
learn = load_learner('/Users/reetvikchatterjee/Desktop/SplitData/train/model.pkl')

# Open the laptop's camera (camera 0 is typically the built-in one)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame from BGR to RGB, then to a PIL image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PILImage.create(rgb_frame)
    
    # Try to make predictions
    try:
        pred, pred_idx, probs = learn.predict(img)
        
        # Display prediction label on the frame
        label = f"{pred}: {probs[pred_idx]:.4f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Prediction error: {e}")
    
    # Display the frame with prediction
    cv2.imshow("Frame", frame)
    
    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
