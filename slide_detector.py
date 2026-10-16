import cv2
import numpy as np
import os

def extract_slide_changes(video_path, threshold=30, skip_seconds=2):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open the video. Check the path:", video_path)
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * skip_seconds) 
    prev_frame = None
    slide_timestamps = []
    frame_count = 0
    
    os.makedirs("slide_frames", exist_ok=True)
    print("Starting slide detection...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))  
            
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                change_score = np.mean(diff)
                
                if change_score > threshold:
                    timestamp = frame_count / fps
                    slide_timestamps.append(round(timestamp, 2))
                
                    filename = f"slide_frames/slide_at_{round(timestamp,2)}s.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Slide change detected at: {round(timestamp,2)} seconds")
            
            prev_frame = gray
        
        frame_count += 1
    
    cap.release()
    print(f"\n✅ Found {len(slide_timestamps)} slide changes in total")
    return slide_timestamps
if __name__ == "__main__":
    video_path = "videos/lecture2.mp4" 
    timestamps =