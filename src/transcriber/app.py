import os
import argparse

import whisper

import numpy as np
import ffmpeg
from pathlib import Path
from openai import OpenAI
from sqlalchemy import create_engine, Column, String, Float
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import sessionmaker , declarative_base
import tempfile
from vtt_utils import merge_webvtt_to_list
from config import *


client = OpenAI(api_key=api_key, base_url=base_url)

# Whisper Setup
model = whisper.load_model("base")

# Database Setup
db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class VideoEmbedding(Base):
    __tablename__ = "video_embeddings"

    id = Column(String, primary_key=True)
    embedding = Column(Vector(2048))
    initial_time = Column(Float)
    title = Column(String)
    thumbnail = Column(String)
    video_url = Column(String)
    text = Column(String)

Base.metadata.create_all(bind=engine)

def store_embedding(video_data):
    session = SessionLocal()
    for entry in video_data:
        embedding_vector=entry["embedding"]
        record = VideoEmbedding(
            id=entry["id"],
            embedding=embedding_vector,
            initial_time=entry["initial_time"],
            title=entry["title"],
            thumbnail=entry["thumbnail"],
            video_url=entry["video_url"],
            text=entry["text"],
        )
        session.add(record)
    session.commit()
    session.close()

# Transcribe audio to text
def transcribe_audio_to_vtt(audio_file_path):
    # Transcribe the audio file
    result = model.transcribe(audio_file_path, verbose=True)
    
    # Initialize the VTT content
    vtt_content = "WEBVTT\n\n"
    
    # Loop through each segment and format it as VTT
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        
        # Convert start and end times to VTT format (HH:MM:SS.MS)
        start_time = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02}.{int((start % 1) * 1000):03}"
        end_time = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02}.{int((end % 1) * 1000):03}"
        
        # Append to the VTT content
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    
    return vtt_content

def process_video(video_url: str) -> dict[str, str]:
    
    video_title = video_url.rsplit("/", 1)[0]
    print(f"Processing video file: {video_title}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_file = os.path.join(tmp_dir, "audio.mp3")
        ffmpeg.input(video_url).output(audio_file, format="mp3", acodec="libmp3lame").run()
        
        whisper_transcript = transcribe_audio_to_vtt(audio_file)

        seconds_to_merge = 8
        transcript = merge_webvtt_to_list(whisper_transcript, seconds_to_merge)
        stride = 3
        video_data = []

        for block in range(0, len(transcript), stride):
            initial_time = transcript[block]["initial_time_in_seconds"]
            
            # Ensure we don't exceed the length of the transcript
            text = " ".join([t["text"] for t in transcript[block : min(block + stride, len(transcript))]]).replace("\n", " ")
            
            # Generate a unique ID
            id = f"video-t{initial_time}-{block}"
            
            video_data.append(
                {
                    "id": id,
                    "initial_time": initial_time,
                    "title": video_title,
                    "thumbnail": "",  # No thumbnail for local files
                    "video_url": f"{video_url}#t={initial_time}", 
                    "text": text,
                    "embedding": client.embeddings.create(
                        input=text, model="nv-embed-1b-v2"
                    ).data[0].embedding,
                }
            )
        
        store_embedding(video_data)
        
        output_transcript_path = Path("transcription.txt")
        with open(output_transcript_path, "w") as transcript_file:
            transcript_text = "\n".join([t["text"] for t in transcript])
            transcript_file.write(transcript_text)
        
        return {"title": "Local Video"}

def main():
    parser = argparse.ArgumentParser(description="Process a local MP4 video file.")
    parser.add_argument("video_path", type=str, help="Path to the local MP4 file")
    args = parser.parse_args()
    
    
    result = process_video(args.video_path)
    print(f"Processing complete: {result}")

if __name__ == "__main__":
    main()
