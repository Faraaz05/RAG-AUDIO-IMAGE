"""
ECS GPU Worker for Audio Processing - Vector Trace
Long-running worker that processes audio/video files from a queue.
"""
import json
import logging
import sys
import os
import subprocess
import tempfile
import shutil
import re
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
import time

from dotenv import load_dotenv
load_dotenv('/app/.env')

# Database & Models
from sqlalchemy import create_engine, Column, Integer, String, Enum, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import enum

# AI & Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

# Audio Processing
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

# Queue backends
import boto3
import redis

# --- QUEUE ABSTRACTIONS ---
class QueueBackend:
    def send(self, payload: dict):
        raise NotImplementedError
    
    def receive(self, max_messages: int = 1):
        raise NotImplementedError
    
    def ack(self, message):
        raise NotImplementedError

class RedisQueue(QueueBackend):
    def __init__(self, host='localhost', port=6379, db=0, queue_name='audio_queue'):
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.queue_name = queue_name
    
    def send(self, payload: dict):
        self.redis.lpush(self.queue_name, json.dumps(payload))
    
    def receive(self, max_messages: int = 1):
        messages = []
        for _ in range(max_messages):
            message = self.redis.brpop(self.queue_name, timeout=1)
            if message:
                messages.append({
                    'body': message[1].decode('utf-8'),
                    'receipt_handle': message[0]  # Use the key as handle
                })
        return messages
    
    def ack(self, message):
        # Redis auto-removes on brpop, no explicit ack needed
        pass

class SQSQueue(QueueBackend):
    def __init__(self, queue_url: str):
        self.sqs = boto3.client('sqs', region_name=os.getenv('AWS_REGION', 'ap-south-1'))
        self.queue_url = queue_url
    
    def send(self, payload: dict):
        self.sqs.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(payload)
        )
    
    def receive(self, max_messages: int = 1):
        response = self.sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            VisibilityTimeout=300  # 5 minutes
        )
        messages = []
        for msg in response.get('Messages', []):
            messages.append({
                'body': msg['Body'],
                'receipt_handle': msg['ReceiptHandle']
            })
        return messages
    
    def ack(self, message):
        self.sqs.delete_message(
            QueueUrl=self.queue_url,
            ReceiptHandle=message['receipt_handle']
        )

def get_queue_backend() -> QueueBackend:
    backend = os.getenv('QUEUE_BACKEND', 'redis')
    if backend == 'redis':
        return RedisQueue(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            queue_name=os.getenv('AUDIO_QUEUE_NAME', 'audio_queue')
        )
    elif backend == 'sqs':
        return SQSQueue(queue_url=os.getenv('AUDIO_SQS_QUEUE_URL'))
    else:
        raise ValueError(f"Unknown QUEUE_BACKEND: {backend}")

# --- MINIMAL MODELS (Bridge to your DB) ---
Base = declarative_base()

class FileStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PARTITIONING = "PARTITIONING"  # Used for both PDF partitioning and audio transcription
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class File(Base):
    __tablename__ = "files"
    file_id = Column(String, primary_key=True)
    project_id = Column(Integer)
    original_filename = Column(String)
    file_path = Column(String)
    status = Column(String)  # Simple string for DB compatibility
    processed_path = Column(String)
    error_message = Column(String)

# --- AUDIO PROCESSING FUNCTIONS ---

def convert_to_wav(input_path: Path, output_path: Path):
    """
    Extract audio (if video) and convert to 16kHz mono WAV for model processing
    """
    log.info(f"🎵 Converting to WAV: {input_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",                 # ignore video stream
        "-ar", "16000",        # 16kHz
        "-ac", "1",            # mono
        str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info(f"✅ WAV created: {output_path.name}")

def compress_audio_for_web(input_path: Path, output_path: Path):
    """
    Create bandwidth-friendly audio file for web streaming (MP3 format, compressed)
    """
    log.info(f"🗜️ Compressing audio for web: {input_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",                 # ignore video stream
        "-ar", "44100",        # Standard quality
        "-ac", "2",            # Stereo
        "-b:a", "128k",        # 128kbps bitrate for streaming
        "-codec:a", "libmp3lame",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info(f"✅ Compressed audio created: {output_path.name}")

def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm for WebVTT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def split_on_speaker_boundaries(segments_list: list, diarization) -> list:
    """Split Whisper segments on diarization speaker boundaries"""
    if not segments_list or diarization is None:
        return segments_list
    
    # Get all speaker turns from diarization
    speaker_turns = []
    for turn, speaker in diarization.speaker_diarization:
        speaker_turns.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': str(speaker)
        })
    
    # Sort by start time
    speaker_turns.sort(key=lambda x: x['start'])
    
    result = []
    
    for seg in segments_list:
        seg_start = seg['start']
        seg_end = seg['end']
        seg_text = seg['text'].strip()
        
        # Find all speaker turns that overlap with this segment
        overlapping_turns = []
        for turn in speaker_turns:
            # Check if turn overlaps with segment
            if turn['end'] > seg_start and turn['start'] < seg_end:
                overlapping_turns.append(turn)
        
        if not overlapping_turns:
            # No speaker info, keep original
            result.append(seg)
            continue
        
        if len(overlapping_turns) == 1:
            # Only one speaker for this segment, assign and keep
            seg['speaker'] = overlapping_turns[0]['speaker']
            result.append(seg)
            continue
        
        # Multiple speakers in this segment - split it
        if seg.get('words'):
            # Use word timestamps for accurate splitting
            for turn in overlapping_turns:
                overlap_start = max(seg_start, turn['start'])
                overlap_end = min(seg_end, turn['end'])
                
                if overlap_end <= overlap_start:
                    continue
                
                # Find words that fall within this speaker turn
                turn_words = []
                for word in seg['words']:
                    word_mid = (word['start'] + word['end']) / 2
                    if overlap_start <= word_mid < overlap_end:
                        turn_words.append(word['word'])
                
                if turn_words:
                    result.append({
                        'start': overlap_start,
                        'end': overlap_end,
                        'text': ''.join(turn_words).strip(),
                        'speaker': turn['speaker'],
                        'words': []
                    })
        else:
            # Fallback: split text proportionally but respect word boundaries
            total_duration = seg_end - seg_start
            words = seg_text.split()
            
            for turn in overlapping_turns:
                overlap_start = max(seg_start, turn['start'])
                overlap_end = min(seg_end, turn['end'])
                
                if overlap_end <= overlap_start:
                    continue
                
                # Calculate which words belong to this turn based on time
                start_ratio = (overlap_start - seg_start) / total_duration
                end_ratio = (overlap_end - seg_start) / total_duration
                
                word_start = int(start_ratio * len(words))
                word_end = int(end_ratio * len(words))
                
                # Ensure we get at least one word if the turn overlaps
                if word_start == word_end and len(words) > 0:
                    word_end = word_start + 1
                
                chunk_words = words[word_start:word_end]
                
                if chunk_words:
                    result.append({
                        'start': overlap_start,
                        'end': overlap_end,
                        'text': ' '.join(chunk_words),
                        'speaker': turn['speaker'],
                        'words': []
                    })
    
    return result

def merge_and_resegment(segments_list: list) -> list:
    """Merge consecutive same-speaker segments and split on sentence boundaries"""
    if not segments_list:
        return []
    
    # First, merge consecutive same-speaker segments
    merged = []
    current = segments_list[0].copy()
    
    for seg in segments_list[1:]:
        if seg['speaker'] == current['speaker']:
            # Same speaker - merge
            current['end'] = seg['end']
            current['text'] += ' ' + seg['text']
            current['words'].extend(seg['words'])
        else:
            # Different speaker - save current and start new
            merged.append(current)
            current = seg.copy()
    merged.append(current)
    
    # Now split on sentence boundaries
    final_segments = []
    for seg in merged:
        text = seg['text'].strip()
        # Split on sentence endings (. ! ? followed by space or end)
        sentences = re.split(r'([.!?])\s+', text)
        
        # Recombine sentences with their punctuation
        complete_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i+1] in '.!?':
                complete_sentences.append(sentences[i] + sentences[i+1])
                i += 2
            elif sentences[i].strip():
                complete_sentences.append(sentences[i])
                i += 1
            else:
                i += 1
        
        if not complete_sentences:
            final_segments.append(seg)
            continue
        
        # If only one sentence, keep original
        if len(complete_sentences) == 1:
            final_segments.append(seg)
            continue
        
        # Split timing proportionally by character count
        total_chars = sum(len(s) for s in complete_sentences)
        duration = seg['end'] - seg['start']
        current_time = seg['start']
        
        for sentence in complete_sentences:
            sent_duration = (len(sentence) / total_chars) * duration
            final_segments.append({
                'start': current_time,
                'end': current_time + sent_duration,
                'text': sentence.strip(),
                'speaker': seg['speaker'],
                'words': []  # Word-level timestamps lost in re-segmentation
            })
            current_time += sent_duration
    
    return final_segments

def transcribe_audio(model: WhisperModel, diarization_pipeline, audio_path: Path):
    """
    Transcribe audio with speaker diarization
    Returns: dict with segments and metadata
    """
    log.info(f"🎤 Transcribing: {audio_path.name}")
    
    # Speaker Diarization
    diarization = None
    if diarization_pipeline:
        log.info("👥 Running speaker diarization...")
        try:
            diarization = diarization_pipeline(str(audio_path))
            # Count speakers
            speakers = set()
            for turn, speaker in diarization.speaker_diarization:
                speakers.add(speaker)
            num_speakers = len(speakers)
            log.info(f"✅ Detected {num_speakers} speakers")
        except Exception as e:
            log.warning(f"⚠️ Diarization failed: {e}. Continuing without speaker labels.")
            diarization = None
    
    # Transcribe with Whisper
    log.info("📝 Running Whisper transcription...")
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True
    )
    
    segments = list(segments)
    log.info(f"✅ Transcribed {len(segments)} segments")
    
    # Build structured output
    raw_segments = [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "speaker": "Speaker",  # Will be assigned by split_on_speaker_boundaries
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end
                } for w in (seg.words or [])
            ]
        }
        for seg in segments
    ]
    
    # Split on speaker boundaries to get correct speaker labels
    speaker_split_segments = split_on_speaker_boundaries(raw_segments, diarization)
    
    # Merge same-speaker segments and re-segment on sentences
    final_segments = merge_and_resegment(speaker_split_segments)
    
    result = {
        "language": info.language,
        "duration": info.duration,
        "text": " ".join(seg["text"] for seg in final_segments),
        "segments": final_segments
    }
    
    log.info(f"✅ Transcription complete: {len(final_segments)} final segments")
    return result

def parse_vtt_to_turns(vtt_text: str) -> List[Dict]:
    """
    Parse VTT transcript into individual speaker turns.
    """
    # Pattern to match: timestamp --> timestamp\nSpeaker: Text
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\s*\n'
        r'(.*?):\s*(.*?)(?=\n\n|\n\d|\Z)',
        re.DOTALL
    )
    
    turns = []
    matches = pattern.findall(vtt_text)
    
    for timestamp, speaker, text in matches:
        turns.append({
            "timestamp": timestamp,
            "speaker": speaker.strip(),
            "text": text.strip()
        })
    
    return turns

def create_speaker_turn_chunks(
    transcript_result: dict,
    audio_name: str,
    audio_date: str,
    project_id: int,
    turns_per_chunk: int = 8,
    overlap: int = 3
) -> List[Dict]:
    """
    Create overlapping chunks from transcript segments based on speaker turns.
    Uses the same logic as transcript.py for consistency.
    """
    log.info(f"🔨 Creating chunks: {audio_name}")
    
    # Convert segments to turns format
    turns = [
        {
            "timestamp": format_timestamp(seg["start"]),
            "speaker": seg["speaker"],
            "text": seg["text"]
        }
        for seg in transcript_result["segments"]
    ]
    
    if not turns:
        log.warning("⚠️ No turns found in transcript")
        return []
    
    log.info(f"✅ Parsed {len(turns)} speaker turns")
    
    # Create overlapping chunks
    chunks = []
    step = max(1, turns_per_chunk - overlap)
    
    for i in range(0, len(turns), step):
        window = turns[i : i + turns_per_chunk]
        
        # Skip very small chunks at the end
        if len(window) < 2:
            break
        
        # Extract metadata
        speakers_list = list(set(t['speaker'] for t in window))
        start_time = window[0]['timestamp']
        end_time = window[-1]['timestamp']
        
        # Combine turn contents
        combined_text = "\n".join([
            f"{t['speaker']}: {t['text']}" for t in window
        ])
        
        # Create enhanced content for better searchability
        enhanced_text = f"""Meeting: {audio_name}
Project ID: {project_id}
Date: {audio_date}
Time Range: {start_time} - {end_time}
Speakers: {', '.join(speakers_list)}

Transcript:
{combined_text}"""
        
        chunks.append({
            "text": combined_text,
            "enhanced_content": enhanced_text,
            "metadata": {
                "source_type": "meeting_transcript",
                "project_id": project_id,
                "meeting_name": audio_name,
                "meeting_date": audio_date,
                "start_time": start_time,
                "end_time": end_time,
                "speakers_in_chunk": json.dumps(speakers_list),
                "turn_count": len(window),
                "chunk_index": len(chunks)
            }
        })
    
    log.info(f"✅ Created {len(chunks)} chunks (turns_per_chunk={turns_per_chunk}, overlap={overlap})")
    return chunks

def export_transcript_to_json(transcript_result: dict, chunks: List[Dict], project_id: int, file_id: str, bucket_name: str, audio_name: str = None, audio_date: str = None):
    """
    Export transcript to JSON in S3 matching the existing transcript schema.
    """
    log.info("📄 Exporting transcript to JSON...")
    
    # Extract unique speakers from segments
    speakers = sorted(list(set(seg.get("speaker", "Unknown") for seg in transcript_result["segments"])))
    
    # Convert segments to turns format with timestamps as HH:MM:SS.mmm
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    turns = [
        {
            "timestamp": format_timestamp(seg["start"]),
            "speaker": seg.get("speaker", "Unknown"),
            "content": seg["text"]
        }
        for seg in transcript_result["segments"]
    ]
    
    # Prepare export data matching the existing transcript structure
    export_data = {
        "meeting_name": audio_name or file_id,
        "meeting_date": audio_date or "",
        "total_turns": len(turns),
        "speakers": speakers,
        "turns": turns,
        "is_audio": True
    }
    
    # Save to S3
    s3_key = f"projects/{project_id}/processed/{file_id}.json"
    json_content = json.dumps(export_data, indent=2, ensure_ascii=False)
    json_bytes = json_content.encode('utf-8')
    
    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json_bytes,
            ContentType='application/json'
        )
        log.info(f"✅ Exported transcript to s3://{bucket_name}/{s3_key}")
        return s3_key
    except Exception as e:
        log.error(f"Failed to save transcript to S3: {e}")
        return None

def upload_compressed_audio_to_s3(audio_path: Path, project_id: int, file_id: str, bucket_name: str):
    """
    Upload compressed audio file to S3.
    """
    log.info(f"☁️ Uploading compressed audio to S3...")
    
    s3_key = f"projects/{project_id}/processed/{file_id}.mp3"
    
    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    try:
        s3.upload_file(
            str(audio_path),
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'audio/mpeg'}
        )
        log.info(f"✅ Uploaded audio to s3://{bucket_name}/{s3_key}")
        return s3_key
    except Exception as e:
        log.error(f"Failed to upload audio to S3: {e}")
        return None

def sanitize_metadata(metadata: dict) -> dict:
    """
    Sanitize metadata to ensure ChromaDB compatibility.
    ChromaDB only accepts strings, ints, floats, and bools.
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = ""
        else:
            # Convert complex objects to JSON strings
            sanitized[key] = json.dumps(value) if not isinstance(value, str) else str(value)
    return sanitized

# --- CONFIG & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("AudioWorker")

# Load DB URL from Env
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Initialize AI Models (loaded once globally)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

log.info(f"🎯 Initializing Whisper model: {WHISPER_MODEL_SIZE} on {DEVICE}")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)
log.info("✅ Whisper model loaded")

# Initialize diarization pipeline
diarization_pipeline = None
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        log.info("👥 Loading speaker diarization pipeline...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        if DEVICE == "cuda":
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
        log.info("✅ Diarization pipeline loaded")
    except Exception as e:
        log.warning(f"⚠️ Failed to load diarization: {e}. Continuing without diarization.")
else:
    log.warning("⚠️ HF_TOKEN not set. Diarization disabled.")

def process_audio_job(job: dict):
    """Process a single audio job from the queue"""
    project_id = job['project_id']
    file_id = job['file_id']
    file_path = job['file_path']
    audio_name = job['audio_name']
    audio_date = job['audio_date']
    bucket_name = os.getenv('S3_BUCKET_NAME')

    db: Session = SessionLocal()
    db_file = db.query(File).filter(File.file_id == file_id).first()
    
    try:
        # Download from S3
        s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        temp_dir = tempfile.mkdtemp()
        
        # Get file extension from file_path
        file_ext = Path(file_path).suffix
        local_input_path = Path(temp_dir) / f"input{file_ext}"
        
        log.info(f"📥 Downloading s3://{bucket_name}/{file_path}")
        s3.download_file(bucket_name, file_path, str(local_input_path))
        
        # Update DB Status (use PARTITIONING for transcription phase)
        if db_file:
            db_file.status = FileStatus.PARTITIONING
            db.commit()
        
        # --- AUDIO PROCESSING PIPELINE ---
        
        # 1. Convert to WAV for model processing
        wav_path = Path(temp_dir) / "audio_16k.wav"
        convert_to_wav(local_input_path, wav_path)
        
        # 2. Transcribe with diarization
        transcript_result = transcribe_audio(whisper_model, diarization_pipeline, wav_path)
        
        # 3. Create chunks (same logic as transcript.py)
        chunks = create_speaker_turn_chunks(
            transcript_result=transcript_result,
            audio_name=audio_name,
            audio_date=audio_date,
            project_id=project_id,
            turns_per_chunk=8,
            overlap=3
        )
        
        # 4. Export transcript and chunks to JSON in S3
        log.info("📄 Exporting transcript to S3...")
        processed_path = export_transcript_to_json(
            transcript_result, chunks, project_id, file_id, bucket_name, audio_name, audio_date
        )
        
        # 5. Create and upload compressed audio for web streaming
        log.info("🗜️ Creating compressed audio...")
        compressed_audio_path = Path(temp_dir) / f"{file_id}.mp3"
        compress_audio_for_web(wav_path, compressed_audio_path)
        
        compressed_s3_path = upload_compressed_audio_to_s3(
            compressed_audio_path, project_id, file_id, bucket_name
        )
        # Note: compressed_audio_path stored in processed_path or separate field if needed
        
        # 6. Embedding
        if db_file:
            db_file.status = FileStatus.EMBEDDING
            db.commit()
        
        log.info("🔮 Generating embeddings...")
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="retrieval_document"
        )
        
        # Extract texts from chunks for embedding
        texts = [chunk['enhanced_content'] for chunk in chunks]
        embeddings = embedding_model.embed_documents(texts)
        
        # 7. ChromaDB Indexing
        if db_file:
            db_file.status = FileStatus.INDEXING
            db.commit()
        
        log.info("💾 Storing in ChromaDB...")
        chroma_client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST"),
            port=int(os.getenv("CHROMA_PORT", 8000))
        )
        collection = chroma_client.get_or_create_collection(name=f"project_{project_id}")
        
        # Prepare data for ChromaDB
        ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [sanitize_metadata(chunk['metadata']) for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        log.info(f"✅ Indexed {len(chunks)} chunks in ChromaDB")
        
        # Finalize
        if db_file:
            db_file.file_path = f"transcript_{audio_name}_{audio_date}"  # Change from raw path
            db_file.processed_path = processed_path
            db_file.status = FileStatus.COMPLETED
            db.commit()
        
        log.info("🎉 Audio processing completed successfully!")
        
    except Exception as e:
        log.error(f"❌ Job Failed: {str(e)}")
        if db_file:
            db_file.status = FileStatus.FAILED
            db_file.error_message = str(e)
            db.commit()
        raise
    finally:
        db.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

def worker_loop():
    """Long-running worker loop that processes audio jobs from the queue"""
    queue = get_queue_backend()
    max_workers = int(os.getenv('MAX_WORKERS', 1))  # Audio processing is more intensive
    
    log.info(f"🚀 Starting audio worker loop with {max_workers} max workers")
    log.info(f"📡 Queue backend: {os.getenv('QUEUE_BACKEND', 'redis')}")
    log.info(f"🎤 Whisper model: {WHISPER_MODEL_SIZE} on {DEVICE}")
    
    while True:
        try:
            # Poll for messages
            messages = queue.receive(max_messages=max_workers)
            
            if not messages:
                log.debug("📭 No messages in queue, sleeping...")
                time.sleep(5)
                continue
            
            log.info(f"📬 Received {len(messages)} messages")
            
            # Process jobs sequentially to avoid GPU model crashes in subprocesses
            for message in messages:
                job = json.loads(message['body'])
                try:
                    process_audio_job(job)
                    queue.ack(message)
                    log.info("✅ Job completed and acknowledged")
                except Exception as e:
                    log.error(f"❌ Job failed, not acknowledging: {str(e)}")
                    # Don't ack failed messages, let them retry
        
        except Exception as e:
            log.error(f"❌ Worker loop error: {str(e)}")
            time.sleep(10)  # Back off on errors

if __name__ == "__main__":
    worker_loop()
