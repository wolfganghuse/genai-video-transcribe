from parliament import Context
from cloudevents.conversion import to_json
from config import *
from urllib.parse import urlparse, urlunparse

import logging
import boto3
import os
import whisper

import ffmpeg
from pathlib import Path
from openai import OpenAI
from sqlalchemy import Index
from sqlalchemy import create_engine, Column, String, Float, Integer
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import sessionmaker , declarative_base
import tempfile
from vtt_utils import merge_webvtt_to_list


client = OpenAI(api_key=api_key, base_url=base_url)

# Whisper Setup
model = whisper.load_model("base")

# Database Setup
engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
Base.metadata.create_all(bind=engine)

class VideoEmbedding(Base):
    __tablename__ = "video_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-increment ID
    embedding = Column(Vector(1024)) 
    initial_time = Column(Float)
    title = Column(String)
    thumbnail = Column(String)
    video_url = Column(String)
    text = Column(String)


# Create an index for cosine distance on the embedding column
Index(
    "ix_video_embeddings_embedding_cosine",  # Index name
    VideoEmbedding.embedding,
    postgresql_using="ivfflat",  # Use the IVFFLAT indexing method
    postgresql_ops={"embedding": "vector_cosine_ops"},  # Use the cosine similarity operator
    postgresql_with={"lists": 100},  # Adjust based on your data
)


def store_embedding(video_data):
    session = SessionLocal()
    
    records = [
        VideoEmbedding(
            embedding=entry["embedding"],
            initial_time=entry["initial_time"],
            title=entry["title"],
            thumbnail=entry["thumbnail"],
            video_url=entry["video_url"],
            text=entry["text"],
        )
        for entry in video_data
    ]
    
    session.bulk_save_objects(records) 
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
    video_title = os.path.basename(video_url).rsplit(".", 1)[0]
    print(f"Processing video file: {video_title}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_file = os.path.join(tmp_dir, "audio.mp3")
        try:
            ffmpeg.input(video_url).output(audio_file, format="mp3", acodec="libmp3lame").run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return {}
        
        whisper_transcript = transcribe_audio_to_vtt(audio_file)
        seconds_to_merge = 6
        transcript = merge_webvtt_to_list(whisper_transcript, seconds_to_merge)
        stride = 2
        
        texts = [
            " ".join([t["text"] for t in transcript[block : min(block + stride, len(transcript))]]).replace("\n", " ")
            for block in range(0, len(transcript), stride)
        ]
        
        embeddings_response = client.embeddings.create(input=texts, model="nv-embed-1b-v2", dimensions=1024)
        embeddings = [item.embedding for item in embeddings_response.data]
        
        video_data = []
        for idx, block in enumerate(range(0, len(transcript), stride)):
            initial_time = transcript[block]["initial_time_in_seconds"]
            id = f"video-t{initial_time}-{block}"
            
            video_data.append(
                {
                    "id": id,
                    "initial_time": initial_time,
                    "title": video_title,
                    "thumbnail": "",
                    "video_url": f"{video_url}#t={initial_time}", 
                    "text": texts[idx],
                    "embedding": embeddings[idx],
                }
            )
        
        store_embedding(video_data)
        output_transcript_path = Path("transcription.txt")
        with open(output_transcript_path, "w") as transcript_file:
            transcript_file.write("\n".join([t["text"] for t in transcript]))
        
        return {"title": video_title}



# Logging Configuration
logger = logging.getLogger('transcribe_backend')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')


# Utility Functions
def remove_query_from_url(url):
    """
    Removes the query parameters from a URL.
    :param url: The original URL
    :return: The URL without the query parameters
    """
    parsed_url = urlparse(url)
    return urlunparse(parsed_url._replace(query=""))

def s3_client():
    """Create and return an S3 client."""
    try:
        session = boto3.Session()
        return session.client('s3',
                              endpoint_url=s3_endpoint_url,
                              aws_access_key_id=access_key,
                              aws_secret_access_key=secret_key,
                              region_name=s3_region,
                              verify=ssl_verify)
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise

def get_signed_url(bucket, obj):
    """
    Generate a presigned URL for accessing an S3 object.
    :param bucket: S3 bucket name
    :param obj: S3 object key
    :return: Presigned URL
    """
    try:
        return s3_client().generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': obj},
            ExpiresIn=signed_url_expiration
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        raise

def main(context: Context):
    """
    Main entry point for processing events.
    :param context: Context containing CloudEvent data
    """
    try:
        source_attributes = context.cloud_event.get_attributes()
        logger.info(f'REQUEST: {to_json(context.cloud_event)}', extra=source_attributes)

        data = context.cloud_event.data
        notification_type = data["Records"][0]["eventName"]

        if notification_type in ["s3:ObjectCreated:Put", "s3:ObjectCreated:CompleteMultipartUpload"]:
            src_bucket = data["Records"][0]["s3"]["bucket"]["name"]
            src_obj = data["Records"][0]["s3"]["object"]["key"]

            signed_url = get_signed_url(src_bucket, src_obj)
            logger.info(f'SIGNED URL: {signed_url}', extra=source_attributes)

            result = process_video(signed_url)
            logger.info(f'Video processed: {result}', extra=source_attributes)
            logger.info(f'Successfully embedded: {notification_type}', extra=source_attributes)

        else:
            logger.info(f'Event not processed: {notification_type}', extra=source_attributes)

    except Exception as e:
        logger.error(f"Failed to process event: {e}")
        return "", 204

    return "", 204
