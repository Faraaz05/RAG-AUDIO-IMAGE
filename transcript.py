"""
Transcript processing service for VTT file ingestion.
Ports logic from transcript_ingestion.ipynb for FastAPI integration.
"""
import re
import json
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from ..core.config import settings
from dotenv import load_dotenv

load_dotenv()

def parse_vtt_to_turns(vtt_text: str) -> List[Dict]:
    """
    Parse VTT transcript into individual speaker turns.
    
    Args:
        vtt_text: Raw VTT transcript text
        
    Returns:
        List of dictionaries containing timestamp, speaker, and text
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
    vtt_text: str,
    meeting_name: str,
    meeting_date: str,
    project_name: str,
    turns_per_chunk: int = 8,
    overlap: int = 3
) -> List[Dict]:
    """
    Create overlapping chunks from VTT transcript based on speaker turns.
    
    Args:
        vtt_text: Raw VTT transcript text
        meeting_name: Name/title of the meeting
        meeting_date: Date of meeting (YYYY-MM-DD format)
        project_name: Name of the project this meeting belongs to
        turns_per_chunk: Number of speaker turns per chunk
        overlap: Number of overlapping turns between chunks
        
    Returns:
        List of chunk dictionaries with text, enhanced_content, and metadata
    """
    print(f"🎙️  Parsing VTT transcript: {meeting_name}")
    
    # Parse VTT into speaker turns
    turns = parse_vtt_to_turns(vtt_text)
    
    if not turns:
        print("⚠️  No speaker turns found in transcript")
        return []
    
    print(f"✅ Parsed {len(turns)} speaker turns")
    
    # Create overlapping chunks
    chunks = []
    step = max(1, turns_per_chunk - overlap)  # Ensure step is at least 1
    
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
        enhanced_text = f"""Meeting: {meeting_name}
Project: {project_name}
Date: {meeting_date}
Time Range: {start_time} - {end_time}
Speakers: {', '.join(speakers_list)}

Transcript:
{combined_text}"""
        
        chunks.append({
            "text": combined_text,
            "enhanced_content": enhanced_text,
            "metadata": {
                "source_type": "meeting_transcript",
                "project_name": project_name,
                "meeting_name": meeting_name,
                "meeting_date": meeting_date,
                "start_time": start_time,
                "end_time": end_time,
                "speakers_in_chunk": json.dumps(speakers_list),  # Serialize list for ChromaDB
                "turn_count": len(window),
                "chunk_index": len(chunks)
            }
        })
    
    print(f"✅ Created {len(chunks)} chunks (turns_per_chunk={turns_per_chunk}, overlap={overlap})")
    return chunks


def format_transcript_for_export(turns: List[Dict], meeting_name: str, meeting_date: str) -> Dict:
    """
    Format parsed transcript turns into JSON structure for frontend rendering.
    
    Args:
        turns: List of parsed speaker turns
        meeting_name: Name of the meeting
        meeting_date: Date of the meeting
        
    Returns:
        Dictionary with meeting metadata and formatted turns
    """
    speakers = list(set(turn['speaker'] for turn in turns))
    
    return {
        "meeting_name": meeting_name,
        "meeting_date": meeting_date,
        "total_turns": len(turns),
        "speakers": speakers,
        "turns": [
            {
                "timestamp": turn['timestamp'],
                "speaker": turn['speaker'],
                "content": turn['text']
            }
            for turn in turns
        ]
    }


def store_transcript_chunks(
    chunks: List[Dict],
    project_id: int,
    embedding_model: GoogleGenerativeAIEmbeddings
) -> int:
    """
    Store transcript chunks in ChromaDB with Google Gemini embeddings.
    
    Args:
        chunks: List of chunk dictionaries from create_speaker_turn_chunks
        project_id: Project ID for collection naming
        embedding_model: Initialized Google Gemini embedding model
        
    Returns:
        Number of chunks stored
    """
    if not chunks:
        print("⚠️  No chunks to store")
        return 0
    
    print(f"💾 Storing {len(chunks)} transcript chunks...")
    
    # Connect to ChromaDB via HttpClient
    chroma_client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT
    )
    
    # Get or create collection for this project
    collection_name = f"project_{project_id}"
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"✅ Using existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": f"Unified document and transcript chunks for project {project_id}"}
        )
        print(f"✅ Created new collection: {collection_name}")
    
    # Prepare data for batch insertion
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        # Generate unique ID
        chunk_id = (
            f"transcript_{chunk['metadata']['meeting_name']}_"
            f"{chunk['metadata']['chunk_index']}"
        ).replace(" ", "_").lower()
        
        ids.append(chunk_id)
        documents.append(chunk['enhanced_content'])
        metadatas.append(chunk['metadata'])
    
    # Generate embeddings in batch (more efficient with Gemini API)
    print("🔮 Generating embeddings via Google Gemini API...")
    batch_embeddings = embedding_model.embed_documents(documents)
    
    # Batch insert into ChromaDB
    collection.add(
        ids=ids,
        embeddings=batch_embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Successfully stored {len(chunks)} transcript chunks")
    print(f"📊 Collection '{collection.name}' now has {collection.count()} total chunks")
    
    return len(chunks)


def process_transcript_file(
    vtt_content: str,
    meeting_name: str,
    meeting_date: str,
    project_id: int,
    project_name: str,
    turns_per_chunk: int = 8,
    overlap: int = 3
) -> Dict:
    """
    Complete transcript processing pipeline: Parse VTT, chunk, embed, and store.
    
    Args:
        vtt_content: Raw VTT transcript content
        meeting_name: Name/title of the meeting
        meeting_date: Date of meeting (YYYY-MM-DD format)
        project_id: Project ID for collection naming
        project_name: Project name for metadata
        turns_per_chunk: Number of speaker turns per chunk
        overlap: Number of overlapping turns between chunks
        
    Returns:
        Dictionary with processing results
    """
    print("=" * 80)
    print("🚀 STARTING TRANSCRIPT INGESTION PIPELINE")
    print("=" * 80)
    
    # Step 1: Create chunks
    print(f"\n🔨 Creating chunks...")
    chunks = create_speaker_turn_chunks(
        vtt_text=vtt_content,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        project_name=project_name,
        turns_per_chunk=turns_per_chunk,
        overlap=overlap
    )
    
    if not chunks:
        return {
            "success": False,
            "error": "No chunks created from transcript",
            "chunks_count": 0
        }
    
    # Step 2: Initialize embedding model
    print(f"\n🔮 Initializing Google Gemini embedding model...")
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_document"
    )
    
    # Step 3: Store in ChromaDB
    print(f"\n💾 Storing chunks in database...")
    chunks_stored = store_transcript_chunks(
        chunks=chunks,
        project_id=project_id,
        embedding_model=embedding_model
    )
    
    # Extract speaker information
    all_speakers = set()
    for chunk in chunks:
        speakers = json.loads(chunk['metadata']['speakers_in_chunk'])
        all_speakers.update(speakers)
    
    print("\n" + "=" * 80)
    print("✅ TRANSCRIPT INGESTION COMPLETE!")
    print("=" * 80)
    
    return {
        "success": True,
        "chunks_count": chunks_stored,
        "meeting_name": meeting_name,
        "meeting_date": meeting_date,
        "speakers": list(all_speakers),
        "collection_name": f"project_{project_id}"
    }
