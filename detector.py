from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("models/best.pt")

def detect_frame(frame, conf=0.75):
    """
    Detects PPE items and estimates counts directly from detected classes,
    since the model does not detect 'Person'.
    """
    original_height, original_width = frame.shape[:2]
    small_frame = cv2.resize(frame, (640, 640))
    results = model(small_frame, conf=conf, verbose=False)[0]
    
    annotated = results.plot()
    annotated = cv2.resize(annotated, (original_width, original_height))

    names = model.names
    classes = results.boxes.cls.cpu().numpy().astype(int)

    # Initialize counters
    helmet_count = 0
    mask_count = 0
    
    # Iterate through all detected boxes and count PPE
    for cls in classes:
        label = names[cls].lower()
        if "helmet" in label or "hardhat" in label:
            helmet_count += 1
        elif "mask" in label:
            mask_count += 1

    # Estimate number of people based on the highest count of PPE detected
    # This ensures that if you have 3 helmets and 3 masks, "People" is 3.
    total_people = max(helmet_count, mask_count)

    # Calculate 'No' counts based on total_people
    no_helmet = max(0, total_people - helmet_count)
    no_mask = max(0, total_people - mask_count)

    # Note: Since we don't have person boxes, individual log tracking is unavailable.
    # We return an empty list for logs to prevent errors.
    stats = {
        "people": total_people,
        "helmet": helmet_count,
        "no_helmet": no_helmet,
        "mask": mask_count,
        "no_mask": no_mask,
        "logs": [] 
    }
    
    return annotated, stats