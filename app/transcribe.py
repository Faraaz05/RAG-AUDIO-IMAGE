import time
from pathlib import Path
import json
import subprocess
import sys
import os

print("[DEBUG] Starting imports...", flush=True)
sys.stdout.flush()

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

print("[DEBUG] All imports loaded successfully", flush=True)
sys.stdout.flush()

# =========================================================
# CONFIG (DO NOT USE RELATIVE PATHS INSIDE DOCKER)
# =========================================================
INPUT_DIR = Path("/data/input")     # mounted from host
OUTPUT_DIR = Path("/data/output")   # mounted from host

MODEL_SIZE = "small"

DEVICE = "cuda"        # "cuda" or "cpu"
COMPUTE_TYPE = "int8"  # best for GTX 1650 and low VRAM

ENABLE_DIARIZATION = True  # Set to False to skip speaker detection
PROCESSED_DIR = INPUT_DIR / "processed"  # Directory to move processed files

# =========================================================
# AUDIO PREPROCESSING
# =========================================================
def convert_to_wav(input_path: Path, output_path: Path):
    """
    Extract audio (if video) and convert to 16kHz mono WAV
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",                 # ignore video stream
        "-ar", "16000",        # 16kHz
        "-ac", "1",            # mono
        str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
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
        # Use word-level timestamps if available, otherwise split by time ratio
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
    import re
    
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

def assign_speaker_to_segment(seg_start: float, seg_end: float, diarization) -> str:
    """Find which speaker was active during a segment"""
    if diarization is None:
        return "Speaker"
    
    # Find overlapping speaker segments using correct pyannote API
    segment_midpoint = (seg_start + seg_end) / 2
    
    # First pass: find speaker at midpoint
    for turn, speaker in diarization.speaker_diarization:
        if turn.start <= segment_midpoint <= turn.end:
            # Speaker is already a string like "SPEAKER_00" from pyannote
            return str(speaker)
    
    # Fallback: find closest speaker
    min_distance = float('inf')
    closest_speaker = "Speaker"
    for turn, speaker in diarization.speaker_diarization:
        distance = min(abs(turn.start - seg_start), abs(turn.end - seg_end))
        if distance < min_distance:
            min_distance = distance
            closest_speaker = str(speaker)
    
    return closest_speaker

# =========================================================
# TRANSCRIPTION
# =========================================================
def transcribe_file(model: WhisperModel, diarization_pipeline, media_path: Path):
    print(f"\n[INFO] Processing: {media_path.name}")

    file_stem = media_path.stem
    file_output_dir = OUTPUT_DIR / file_stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = file_output_dir / "audio_16k.wav"
    json_path = file_output_dir / "transcript.json"
    text_path = file_output_dir / "transcript.txt"
    vtt_path = file_output_dir / "transcript.vtt"

    # 1️⃣ Convert to WAV
    print("[INFO] Converting to WAV...")
    convert_to_wav(media_path, wav_path)

    # 2️⃣ Speaker Diarization (if enabled)
    diarization = None
    if ENABLE_DIARIZATION and diarization_pipeline:
        print("[INFO] Running speaker diarization...")
        try:
            diarization = diarization_pipeline(str(wav_path))
            # Count speakers using correct API
            speakers = set()
            for turn, speaker in diarization.speaker_diarization:
                speakers.add(speaker)
            num_speakers = len(speakers)
            print(f"[INFO] Detected {num_speakers} speakers")
        except Exception as e:
            print(f"[WARN] Diarization failed: {e}. Continuing without speaker labels.")
            diarization = None

    # 3️⃣ Transcribe
    print("[INFO] Transcribing...")
    segments, info = model.transcribe(
        str(wav_path),
        language="en",  # Force English to prevent hallucinations
        word_timestamps=True
    )

    segments = list(segments)

    # 4️⃣ Build structured output
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
    
    # First split on speaker boundaries to get correct speaker labels
    speaker_split_segments = split_on_speaker_boundaries(raw_segments, diarization)
    
    # Then merge same-speaker segments and re-segment on sentences
    final_segments = merge_and_resegment(speaker_split_segments)
    
    result = {
        "source_file": media_path.name,
        "language": info.language,
        "duration": info.duration,
        "text": " ".join(seg["text"] for seg in final_segments),
        "segments": final_segments
    }

    # 5️⃣ Save JSON output
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 6️⃣ Save plain text
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    # 7️⃣ Generate WebVTT format
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, seg in enumerate(result["segments"], start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    # 8️⃣ Print preview
    print("[INFO] Segments:")
    for seg in result["segments"][:5]:
        print(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['speaker']}: {seg['text']}")

    print(f"[INFO] Saved to: {file_output_dir}")
    print(f"[INFO] WebVTT saved: {vtt_path.name}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("[INFO] Starting transcription service")

    if not INPUT_DIR.exists():
        raise RuntimeError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("[INFO] Loading faster-whisper model...", flush=True)
    print(f"[INFO] Device: {DEVICE}, Compute Type: {COMPUTE_TYPE}", flush=True)
    sys.stdout.flush()
    
    try:
        import ctranslate2
        print(f"[INFO] CTranslate2 version: {ctranslate2.__version__}", flush=True)
        print(f"[INFO] CUDA available: {ctranslate2.get_cuda_device_count() > 0}", flush=True)
        print(f"[INFO] CUDA device count: {ctranslate2.get_cuda_device_count()}", flush=True)
    except Exception as e:
        print(f"[WARN] Could not check CUDA: {e}", flush=True)
    
    sys.stdout.flush()
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    print("[INFO] Model loaded successfully!", flush=True)
    sys.stdout.flush()

    # Load diarization pipeline
    diarization_pipeline = None
    if ENABLE_DIARIZATION:
        print("[INFO] Loading speaker diarization pipeline...", flush=True)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("[WARN] HF_TOKEN not set. Diarization requires HuggingFace token.")
            print("[WARN] Visit: https://huggingface.co/pyannote/speaker-diarization")
            print("[WARN] Continuing without diarization...")
        else:
            try:
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
                if DEVICE == "cuda" and torch.cuda.is_available():
                    diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
                print("[INFO] Diarization pipeline loaded successfully!", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to load diarization: {e}")
                print("[WARN] Continuing without diarization...")
    else:
        print("[INFO] Diarization disabled in config", flush=True)

    cache_dir = Path.home() / ".cache" / "ctranslate2"
    print(f"[INFO] Model cache directory: {cache_dir}")
    if cache_dir.exists():
        print(f"[INFO] Cache contents: {list(cache_dir.iterdir()) if cache_dir.is_dir() else 'Not a dir'}")
    else:
        print("[INFO] Cache directory does not exist yet.")

    print("[INFO] Watching for new files in input directory...")

    processed_files = set()

    while True:
        # Get current files
        current_files = set(p for p in INPUT_DIR.iterdir() if p.is_file())

        # Find new files
        new_files = current_files - processed_files

        for media in new_files:
            try:
                transcribe_file(model, diarization_pipeline, media)
                processed_files.add(media)
            except Exception as e:
                print(f"[ERROR] Failed to process {media.name}: {e}")

        # Wait before checking again
        time.sleep(5)


if __name__ == "__main__":
    main()
