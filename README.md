# RAG Audio Transcription Worker

## Overview

This repository contains the GPU-accelerated audio transcription microservice for the RAG FastAPI platform. It operates as a containerized worker process that consumes audio/video processing jobs from distributed queues, performs state-of-the-art speech recognition with speaker diarization, and indexes transcripts into vector databases for semantic search and retrieval.

**Main Project Repository**: [rag-fastapi](https://github.com/Faraaz05/rag-fastapi)

## System Architecture

### Architecture Overview

The audio worker implements a queue-based asynchronous processing architecture designed for horizontal scalability and fault tolerance:

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   API Server    │─Queue──▶│  Audio Worker    │────────▶│   ChromaDB      │
│  (rag-fastapi)  │         │  (This Service)  │         │ Vector Store    │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │                             ▲
                                     ▼                             │
                            ┌─────────────────┐                   │
                            │   PostgreSQL    │                   │
                            │   (Metadata)    │                   │
                            └─────────────────┘                   │
                                     │                             │
                                     ▼                             │
                            ┌─────────────────┐                   │
                            │   Amazon S3     │───────────────────┘
                            │ (Raw + Processed)│
                            └─────────────────┘
```

### Processing Pipeline

1. **Message Reception**: Worker polls Redis/SQS queue for audio processing jobs
2. **File Acquisition**: Downloads audio/video files from S3 storage
3. **Audio Preprocessing**: Converts media to 16kHz mono WAV format using FFmpeg
4. **Speaker Diarization**: Segments audio by speaker using Pyannote neural models
5. **Speech Recognition**: Transcribes speech to text using Faster-Whisper (CTranslate2)
6. **Turn Segmentation**: Aligns transcription with speaker boundaries and re-segments on sentence breaks
7. **Chunk Creation**: Generates overlapping conversation chunks for retrieval context
8. **Embedding Generation**: Creates vector embeddings using Google Gemini Embedding Model
9. **Vector Indexing**: Stores embeddings and metadata in ChromaDB for semantic search
10. **Artifact Storage**: Uploads WebVTT transcript, JSON data, and compressed audio to S3
11. **Status Tracking**: Updates PostgreSQL database with processing state and results

## Key Features

### Advanced Audio Processing

- **Multi-Format Support**: Processes audio (MP3, WAV, FLAC, M4A) and video (MP4, AVI, MOV) formats
- **Automatic Conversion**: FFmpeg-based pipeline for format normalization and audio extraction
- **Web-Optimized Output**: Generates compressed MP3 streams (128kbps) for browser playback

### State-of-the-Art Speech Recognition

- **Faster-Whisper Integration**: CTranslate2-optimized Whisper models for 4x inference speedup
- **GPU Acceleration**: CUDA-enabled transcription on NVIDIA GPUs
- **Word-Level Timestamps**: Precise alignment for time-accurate retrieval
- **Multi-Language Support**: Automatic language detection and transcription

### Speaker Diarization

- **Pyannote Audio Pipeline**: Neural diarization using pretrained segmentation and speaker embedding models
- **Speaker Boundary Splitting**: Intelligent segmentation that respects speaker turn changes
- **Multi-Speaker Transcripts**: Clear attribution of dialogue to individual speakers
- **Overlap Handling**: Robust processing of overlapping speech segments

### Intelligent Chunking Strategy

- **Turn-Based Chunking**: Groups consecutive speaker turns into contextual windows
- **Configurable Overlap**: Sliding window approach with adjustable overlap for context retention
- **Metadata Enrichment**: Each chunk includes timestamps, speaker lists, and meeting context
- **Sentence Boundary Awareness**: Merges and splits segments to preserve linguistic coherence

### Vector Search Integration

- **Semantic Embeddings**: Google Gemini text-embedding-001 model for retrieval-optimized vectors
- **ChromaDB Storage**: High-performance vector database with HTTP API support
- **Project Isolation**: Separate collections per project for multi-tenancy
- **Rich Metadata**: Searchable speaker, timestamp, and meeting information

### Enterprise-Grade Queue Architecture

- **Queue Backend Abstraction**: Pluggable queue interface supporting Redis and Amazon SQS
- **Fault Tolerance**: Visibility timeout and message acknowledgment for reliable processing
- **Horizontal Scalability**: Multiple worker instances for parallel job processing
- **Dead Letter Handling**: Failed jobs remain visible for retry or manual intervention

### Cloud-Native Deployment

- **Docker Containerization**: NVIDIA CUDA base image with GPU runtime support
- **AWS CodeBuild Integration**: Automated CI/CD pipeline with buildspec configuration
- **Amazon ECS Deployment**: GPU-enabled container orchestration on EC2 instances
- **Environment-Based Configuration**: 12-factor application design with .env support

## Technical Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Speech Recognition | Faster-Whisper (CTranslate2) | Efficient Whisper model inference |
| Speaker Diarization | Pyannote Audio 3.1 | Neural speaker segmentation |
| Vector Embeddings | Google Gemini Embedding | Semantic text representation |
| Vector Database | ChromaDB | Persistent vector storage and retrieval |
| Queue Systems | Redis / Amazon SQS | Asynchronous job distribution |
| Object Storage | Amazon S3 | File and artifact storage |
| Relational Database | PostgreSQL | Metadata and state persistence |
| Audio Processing | FFmpeg | Media format conversion |
| Deep Learning | PyTorch + CUDA 12.2 | GPU-accelerated model inference |
| Container Runtime | Docker (NVIDIA Runtime) | Isolated execution environment |

### Python Dependencies

```
faster-whisper      # Optimized Whisper inference
pyannote.audio      # Speaker diarization
torch / torchaudio  # Deep learning framework
chromadb            # Vector database client
langchain_google_genai  # Google Gemini embeddings
sqlalchemy          # ORM for database operations
boto3               # AWS SDK for Python
redis               # Redis client
ffmpeg-python       # Audio processing
pydub               # Audio manipulation
psycopg2-binary     # PostgreSQL adapter
python-dotenv       # Environment variable management
```

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.5+ (recommended: RTX 3060 or higher)
- **VRAM**: Minimum 6GB for small model, 8GB+ recommended for large models
- **RAM**: 16GB minimum, 32GB recommended for concurrent processing
- **Storage**: 20GB for model cache, additional space for temporary processing

### Software Requirements

- **OS**: Linux (Ubuntu 22.04 LTS recommended)
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **NVIDIA Driver**: 525+ (CUDA 12.2 compatible)
- **CUDA**: 12.2.0 runtime (included in Docker image)

### Cloud Environment

- **AWS ECS**: GPU-optimized EC2 instances (g4dn, p3, or p4d families)
- **IAM Permissions**: S3 read/write, SQS receive/delete, ECR pull
- **Network**: VPC with internet gateway for Hugging Face model downloads

## Installation and Setup

### Local Development Setup

#### 1. Clone Repository

```bash
git clone https://github.com/Faraaz05/RAG-AUDIO-TRANSCRIBE.git
cd RAG-AUDIO-TRANSCRIBE
```

#### 2. Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 3. Download Pretrained Models

Models are automatically downloaded from Hugging Face on first run and cached in `model_cache/`. To pre-download:

```bash
# Create cache directory
mkdir -p model_cache/hub

# Models will be downloaded automatically on first execution:
# - pyannote/segmentation-3.0
# - pyannote/speaker-diarization-3.1
# - pyannote/wespeaker-voxceleb-resnet34-LM
# - Systran/faster-whisper-small (or tiny/medium/large)
```

**Note**: Pyannote models require Hugging Face authentication. Set `HF_TOKEN` environment variable with a valid access token.

#### 4. Configure Environment Variables

Create `.env` file in the project root:

```bash
# Queue Configuration
QUEUE_BACKEND=redis  # Options: redis, sqs
REDIS_HOST=localhost
REDIS_PORT=6379
AUDIO_QUEUE_NAME=audio_queue

# AWS Configuration (if using SQS)
AWS_REGION=ap-south-1
AUDIO_SQS_QUEUE_URL=https://sqs.ap-south-1.amazonaws.com/123456789/audio-queue
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
BUCKET_NAME=your-s3-bucket

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Embedding Configuration
GOOGLE_API_KEY=your_gemini_api_key

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token

# Model Configuration
WHISPER_MODEL_SIZE=small  # Options: tiny, small, medium, large
DEVICE=cuda  # Options: cuda, cpu
MAX_WORKERS=1  # Number of concurrent jobs (GPU memory dependent)
```

#### 5. Build Docker Image

```bash
docker build -t faster-whisper-image .
```

#### 6. Run Worker Container

```bash
docker compose up
```

### AWS Deployment

#### 1. ECR Repository Setup

```bash
# Create ECR repository
aws ecr create-repository --repository-name rag-audio-worker --region ap-south-1

# Authenticate Docker to ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-south-1.amazonaws.com
```

#### 2. CodeBuild Configuration

The included `buildspec.yml` provides automated CI/CD:

- Builds Docker image on code push
- Tags with commit hash and `latest`
- Pushes to Amazon ECR
- Generates ECS task definition

Set CodeBuild environment variables:
- `AWS_ACCOUNT_ID`
- `AWS_DEFAULT_REGION`
- `IMAGE_REPO_NAME`

#### 3. ECS Task Definition

Create ECS task definition with:

```json
{
  "family": "rag-audio-worker",
  "requiresCompatibilities": ["EC2"],
  "networkMode": "host",
  "containerDefinitions": [{
    "name": "audio-worker",
    "image": "<account-id>.dkr.ecr.ap-south-1.amazonaws.com/rag-audio-worker:latest",
    "resourceRequirements": [
      {"type": "GPU", "value": "1"}
    ],
    "environment": [
      {"name": "QUEUE_BACKEND", "value": "sqs"},
      {"name": "WHISPER_MODEL_SIZE", "value": "small"}
      // ... additional environment variables
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/rag-audio-worker",
        "awslogs-region": "ap-south-1",
        "awslogs-stream-prefix": "worker"
      }
    }
  }]
}
```

#### 4. ECS Service Deployment

```bash
aws ecs create-service \
  --cluster rag-cluster \
  --service-name audio-worker-service \
  --task-definition rag-audio-worker \
  --desired-count 2 \
  --launch-type EC2 \
  --placement-constraints type="memberOf",expression="attribute:ecs.instance-type =~ g4dn.*"
```

## Configuration

### Model Selection

Configure transcription model size in `.env`:

| Model | Parameters | VRAM | Speed | Accuracy |
|-------|-----------|------|-------|----------|
| tiny | 39M | 1GB | Fastest | Basic |
| small | 244M | 2GB | Fast | Good |
| medium | 769M | 5GB | Moderate | Better |
| large | 1550M | 10GB | Slow | Best |

**Recommendation**: Use `small` for real-time applications, `medium` or `large` for maximum accuracy.

### Chunking Parameters

Modify in `create_speaker_turn_chunks()` function:

```python
turns_per_chunk = 8   # Number of speaker turns per retrieval chunk
overlap = 3           # Turn overlap between adjacent chunks
```

**Guidelines**:
- Larger chunks: Better context, less granular search
- Smaller chunks: More precise retrieval, less context
- Overlap: Prevents information loss at boundaries

### Queue Backend Selection

**Redis**: Low latency, simple setup, suitable for single-region deployments

```bash
QUEUE_BACKEND=redis
REDIS_HOST=localhost
```

**Amazon SQS**: Managed service, multi-region, enterprise-grade reliability

```bash
QUEUE_BACKEND=sqs
AUDIO_SQS_QUEUE_URL=https://sqs.region.amazonaws.com/account/queue-name
```

## Usage

### Job Message Format

Queue messages must conform to the following JSON schema:

```json
{
  "project_id": 123,
  "file_id": "uuid-string",
  "s3_path": "raw/project_123/audio_file.mp4",
  "bucket_name": "rag-bucket",
  "audio_name": "team_meeting",
  "audio_date": "2026-03-08"
}
```

### API Integration Example

From the main FastAPI application (see [rag-fastapi](https://github.com/Faraaz05/rag-fastapi)):

```python
import boto3
import json

# Send job to SQS
sqs = boto3.client('sqs', region_name='ap-south-1')
response = sqs.send_message(
    QueueUrl=os.getenv('AUDIO_SQS_QUEUE_URL'),
    MessageBody=json.dumps({
        'project_id': project.id,
        'file_id': str(file.file_id),
        's3_path': s3_key,
        'bucket_name': os.getenv('BUCKET_NAME'),
        'audio_name': sanitized_filename,
        'audio_date': datetime.utcnow().strftime('%Y-%m-%d')
    })
)
```

### Output Artifacts

After successful processing, the following artifacts are available:

#### S3 Structure

```
processed/
└── project_{project_id}/
    ├── transcript_{audio_name}_{date}.json       # Full transcript + chunks
    ├── transcript_{audio_name}_{date}.vtt        # WebVTT subtitle file
    └── compressed_audio/
        └── {file_id}.mp3                          # Web-optimized audio
```

#### JSON Output Format

```json
{
  "transcript": {
    "language": "en",
    "duration": 1234.56,
    "text": "Full transcript text...",
    "segments": [
      {
        "start": 0.0,
        "end": 5.32,
        "text": "Hello everyone",
        "speaker": "SPEAKER_00"
      }
    ]
  },
  "chunks": [
    {
      "text": "SPEAKER_00: Hello...\nSPEAKER_01: Hi...",
      "enhanced_content": "Meeting: team_meeting...",
      "metadata": {
        "source": "team_meeting",
        "timestamp_start": "00:00:15.000",
        "timestamp_end": "00:02:30.000",
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "project_id": 123,
        "file_id": "uuid",
        "chunk_index": 0
      }
    }
  ]
}
```

#### ChromaDB Storage

Each chunk is stored with:
- **Document**: Raw conversation text
- **Embedding**: 768-dimensional vector (Gemini embedding-001)
- **Metadata**: Speakers, timestamps, project context
- **ID**: `{file_id}_chunk_{index}`

# View logs in Docker
docker compose logs -f transcriber

# View logs in ECS
aws logs tail /ecs/rag-audio-worker --follow
```

## Performance Optimization

### GPU Memory Management

For concurrent processing, estimate memory requirements:

```
Memory = Model_Size + (Audio_Duration * 0.05GB)

Example (small model, 60min audio):
6GB = 2GB_model + (60 * 0.05GB_per_minute) + 1GB_buffer
```

### Processing Speed Benchmarks

| Model | GPU | Audio Duration | Processing Time | Real-time Factor |
|-------|-----|----------------|-----------------|------------------|
| tiny | RTX 3060 | 60 min | 2 min | 30x |
| small | RTX 3060 | 60 min | 4 min | 15x |
| medium | RTX 3090 | 60 min | 8 min | 7.5x |
| large | A100 | 60 min | 12 min | 5x |

### Optimization Recommendations

1. **Model Caching**: Mount persistent volume to avoid re-downloading models
2. **Batch Size**: Set `MAX_WORKERS=1` for GPU-constrained environments
3. **Queue Prefetch**: Increase SQS `MaxNumberOfMessages` for better throughput
4. **S3 Transfer**: Use S3 Transfer Acceleration for faster downloads
5. **CPU Fallback**: Set `DEVICE=cpu` if GPU unavailable (slower but functional)

## Development Workflow

### Local Testing

```bash
# Start dependencies
docker compose up -d redis postgres chroma

# Run worker locally
python3 aws_audio_worker.py

# Send test job to Redis
redis-cli LPUSH audio_queue '{"project_id": 1, "file_id": "test", ...}'
```

### Debugging

Enable verbose logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

Test individual components:

```python
from aws_audio_worker import transcribe_with_diarization

result = transcribe_with_diarization(
    audio_path="test.wav",
    whisper_model=model,
    diarization_pipeline=pipeline
)
print(json.dumps(result, indent=2))
```

### Code Organization

```
aws_audio_worker.py
├── QueueBackend (Abstract)
│   ├── RedisQueue
│   └── SQSQueue
├── Database Models
│   ├── File
│   └── FileStatus
├── Audio Processing
│   ├── convert_to_wav()
│   ├── compress_audio_for_web()
│   └── format_timestamp()
├── Transcription & Diarization
│   ├── transcribe_with_diarization()
│   ├── split_on_speaker_boundaries()
│   └── merge_and_resegment()
├── Chunking & Export
│   ├── create_speaker_turn_chunks()
│   ├── export_vtt()
│   └── export_transcript_to_json()
├── Embedding & Indexing
│   └── process_audio_job()
└── Worker Loop
    └── worker_loop()
```

## Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`
- **Solution**: Reduce `WHISPER_MODEL_SIZE` or decrease concurrent workers

**Issue**: `Model not found on Hugging Face Hub`
- **Solution**: Verify `HF_TOKEN` is set and valid for Pyannote models

**Issue**: `chromadb.errors.InvalidCollectionException`
- **Solution**: Ensure ChromaDB server is running and accessible

**Issue**: `S3 access denied`
- **Solution**: Verify IAM role/credentials have `s3:GetObject` and `s3:PutObject` permissions

**Issue**: `Queue connection timeout`
- **Solution**: Check network connectivity and queue URL/credentials

### Health Checks

Verify system components:

```bash
# GPU availability
nvidia-smi

# Redis connectivity
redis-cli ping

# ChromaDB health
curl http://localhost:8000/api/v1/heartbeat

# S3 access
aws s3 ls s3://your-bucket/

# Database connection
psql -h localhost -U user -d rag_db -c "SELECT 1;"
```

## License

This project is part of the RAG FastAPI platform. See the main repository for license information.

## Acknowledgments

This project leverages the following open-source technologies:

- **Faster-Whisper**: CTranslate2 optimized Whisper models
- **Pyannote Audio**: Speaker diarization toolkit
- **ChromaDB**: Embedding database for vector search
- **FFmpeg**: Multimedia processing framework

## Contact

**Project Maintainer**: Faraaz Bhojawala

**Email**: bhojawalafaraaz@gmail.com

**LinkedIn**: [https://www.linkedin.com/in/faraaz-bhojawala/](https://www.linkedin.com/in/faraaz-bhojawala/)

**GitHub**: [https://github.com/Faraaz05](https://github.com/Faraaz05)

---

**Related Projects**:
- Main Platform: [rag-fastapi](https://github.com/Faraaz05/rag-fastapi)
- Audio Worker: This repository

For questions, issues, or collaboration opportunities, please open an issue or reach out directly.
