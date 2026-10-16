import whisper

print("Loading model...")
model = whisper.load_model("base")

video_path = "videos/lecture2.mp4"

print("Starting transcription...")
result = model.transcribe(video_path)

print("\n--- TRANSCRIPT ---\n")
print(result["text"])

with open("transcripts/lecture1_transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("\n✅ Transcript saved: transcripts/lecture1_transcript.txt")