#!/usr/bin/env python3
"""Search, download, VLM-annotate, filter and export YouTube escalation videos for the anticipation dataset.

Usage:
    python setup/curate_youtube.py --output_dir data/anticipation_curation search
    python setup/curate_youtube.py download --max_downloads 50
    python setup/curate_youtube.py analyze --device cuda:0
    python setup/curate_youtube.py filter --min_anticipation 10
    python setup/curate_youtube.py --output_dir data/anticipation_dataset export
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import csv

try:
    import cv2
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install opencv-python pillow tqdm numpy")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class VideoCandidate:
    video_id: str
    source: str  # "youtube", "wikimedia", "local"
    url: str
    title: str
    duration_seconds: float
    local_path: Optional[str] = None
    download_status: str = "pending"  # pending, downloaded, failed, skipped
    analysis_status: str = "pending"  # pending, analyzed, failed
    
    has_violence: bool = False
    violence_start_frame: Optional[int] = None
    violence_start_second: Optional[float] = None
    escalation_start_frame: Optional[int] = None
    escalation_start_second: Optional[float] = None
    anticipation_window_seconds: Optional[float] = None
    stage_timeline: Optional[List[Dict]] = None
    quality_score: Optional[float] = None  # 0-1
    notes: str = ""


SEARCH_QUERIES = [
    "road rage fight full video",
    "road rage confrontation turns violent",
    "road rage argument to fight",
    "dashcam road rage fight",
    
    "bodycam arrest turns violent",
    "bodycam confrontation fight",
    "police bodycam altercation",
    "bodycam use of force full video",
    
    "street argument turns into fight",
    "verbal altercation becomes physical",
    "confrontation escalates to violence",
    
    "security camera fight full video",
    "cctv confrontation",
    "bar fight security camera",
    "club fight cctv",
    
    "customer attacks employee full video",
    "parking lot confrontation fight",
    "neighbor dispute turns violent",
]

# Title keywords: GOOD hints at unedited footage with the lead-up, BAD at clipped compilations.
GOOD_INDICATORS = [
    "full video", "unedited", "complete", "entire incident",
    "starts with", "argument", "confrontation", "altercation",
    "escalates", "turns violent", "before the fight"
]

BAD_INDICATORS = [
    "compilation", "best of", "top 10", "knockout", "instant karma",
    "fight only", "action only", "highlights"
]


def parse_duration(duration_str: str) -> int:
    """Parse ISO 8601 duration (PT1H2M3S) to seconds."""
    import re
    if not duration_str:
        return 0
    
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


def search_youtube_api(query: str, api_key: str, max_results: int = 10) -> List[Dict]:
    """Search YouTube with the Data API v3."""
    import urllib.request
    import urllib.parse
    
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "key": api_key,
        "videoDuration": "medium",  # 4-20 minutes
        "safeSearch": "none",
    }
    
    try:
        url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
        with urllib.request.urlopen(url, timeout=30) as response:
            search_data = json.loads(response.read().decode())
    except Exception as e:
        print(f"  API search error: {e}")
        return []
    
    video_ids = [item["id"]["videoId"] for item in search_data.get("items", [])]
    
    if not video_ids:
        return []
    
    details_url = "https://www.googleapis.com/youtube/v3/videos"
    details_params = {
        "part": "contentDetails,snippet",
        "id": ",".join(video_ids),
        "key": api_key,
    }
    
    try:
        url = f"{details_url}?{urllib.parse.urlencode(details_params)}"
        with urllib.request.urlopen(url, timeout=30) as response:
            details_data = json.loads(response.read().decode())
    except Exception as e:
        print(f"  API details error: {e}")
        return []
    
    videos = []
    for item in details_data.get("items", []):
        video_id = item["id"]
        snippet = item.get("snippet", {})
        content = item.get("contentDetails", {})
        
        duration = parse_duration(content.get("duration", ""))
        
        videos.append({
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": duration,
            "source": "youtube",
            "description": snippet.get("description", "")[:500],
            "channel": snippet.get("channelTitle", ""),
        })
    
    return videos


def search_youtube_ytdlp(query: str, max_results: int = 10) -> List[Dict]:
    """Search YouTube with yt-dlp (no API key needed, slower and may be rate-limited)."""
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--ignore-errors",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    videos.append({
                        "video_id": data.get("id", ""),
                        "title": data.get("title", ""),
                        "url": data.get("url", f"https://www.youtube.com/watch?v={data.get('id', '')}"),
                        "duration": data.get("duration", 0),
                        "source": "youtube",
                    })
                except json.JSONDecodeError:
                    continue
        return videos
    except subprocess.TimeoutExpired:
        print(f"  Timeout searching for: {query}")
        return []
    except Exception as e:
        print(f"  Error searching: {e}")
        return []


def search_youtube(query: str, max_results: int = 10, api_key: str = None) -> List[Dict]:
    if api_key:
        return search_youtube_api(query, api_key, max_results)
    else:
        return search_youtube_ytdlp(query, max_results)


def filter_candidates(videos: List[Dict], 
                     min_duration: int = 30,
                     max_duration: int = 600) -> List[Dict]:
    filtered = []
    
    for v in videos:
        duration = v.get("duration", 0)
        title = v.get("title", "").lower()
        
        if duration < min_duration or duration > max_duration:
            continue
        
        if any(bad in title for bad in BAD_INDICATORS):
            continue
        
        good_score = sum(1 for good in GOOD_INDICATORS if good in title)
        v["quality_hint"] = good_score
        
        filtered.append(v)
    
    filtered.sort(key=lambda x: x.get("quality_hint", 0), reverse=True)
    
    return filtered


def download_video(url: str, output_dir: str, video_id: str) -> Optional[str]:
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    if os.path.exists(output_path):
        return output_path
    
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]",  # Limit resolution for faster processing
        "-o", output_path,
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if os.path.exists(output_path):
            return output_path
        for f in os.listdir(output_dir):
            if f.startswith(video_id):
                return os.path.join(output_dir, f)
        return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout downloading: {video_id}")
        return None
    except Exception as e:
        print(f"  Error downloading {video_id}: {e}")
        return None


def extract_frames(video_path: str, 
                   fps: float = 1.0,
                   max_frames: int = 300) -> List[Tuple[int, float, Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append((frame_idx, timestamp, pil_image))
            
            if len(frames) >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    return frames


def analyze_video_with_vlm(video_path: str,
                           teacher,
                           fps: float = 1.0,
                           verbose: bool = False) -> Dict:
    """Run the VLM frame by frame; return the stage timeline, violence/escalation onsets and the anticipation window (s)."""
    from vlm.qwen_teacher import simple_to_vector
    
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return {"error": "Could not extract frames"}
    
    if verbose:
        print(f"  Extracted {len(frames)} frames")
    
    teacher.reset_context()
    
    stage_timeline = []
    
    for frame_idx, timestamp, image in tqdm(frames, desc="Analyzing", disable=not verbose):
        try:
            parsed, raw_text = teacher.infer_one(
                image,
                return_embedding=False,
                t=frame_idx,
                raw_frame_idx=frame_idx,
            )
            
            if parsed:
                stage = parsed.get("stage", "unknown")
                overall_threat = parsed.get("overall_threat", 0.0)
                
                stage_timeline.append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "stage": stage,
                    "overall_threat": overall_threat,
                    "confrontation": parsed.get("confrontation_score", 0.0),
                    "aggression": parsed.get("aggressive_posture_score", 0.0),
                    "physical_contact": parsed.get("physical_contact_score", 0.0),
                    "indicators": parsed.get("indicators", []),
                })
        except Exception as e:
            if verbose:
                print(f"  Error at frame {frame_idx}: {e}")
            continue
    
    if not stage_timeline:
        return {"error": "No frames analyzed successfully"}
    
    violence_start = None
    for entry in stage_timeline:
        if entry["stage"] == "violence":
            violence_start = entry["timestamp"]
            break
    
    escalation_start = None
    for entry in stage_timeline:
        if entry["stage"] in ["tension", "escalation"]:
            escalation_start = entry["timestamp"]
            break
    
    anticipation_window = None
    if violence_start is not None and escalation_start is not None:
        if escalation_start < violence_start:
            anticipation_window = violence_start - escalation_start
    
    return {
        "stage_timeline": stage_timeline,
        "violence_start": violence_start,
        "escalation_start": escalation_start,
        "anticipation_window": anticipation_window,
        "total_frames_analyzed": len(stage_timeline),
        "has_violence": violence_start is not None,
        "has_escalation": escalation_start is not None,
    }


def cmd_search(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    candidates_file = output_dir / "candidates.json"
    
    existing = {}
    if candidates_file.exists():
        with open(candidates_file) as f:
            data = json.load(f)
            existing = {c["video_id"]: c for c in data.get("candidates", [])}
    
    api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if api_key:
        print(f"Using YouTube Data API v3")
    else:
        print(f"No API key provided, using yt-dlp (slower)")
    
    print(f"Searching for videos with {len(SEARCH_QUERIES)} queries...")
    
    all_candidates = []
    seen_ids = set(existing.keys())
    
    for query in tqdm(SEARCH_QUERIES, desc="Searching"):
        results = search_youtube(query, max_results=args.max_per_query, api_key=api_key)
        filtered = filter_candidates(
            results,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        
        for v in filtered:
            if v["video_id"] not in seen_ids:
                seen_ids.add(v["video_id"])
                candidate = VideoCandidate(
                    video_id=v["video_id"],
                    source=v["source"],
                    url=v["url"],
                    title=v["title"],
                    duration_seconds=v.get("duration", 0),
                )
                all_candidates.append(asdict(candidate))
    
    for vid, c in existing.items():
        if vid not in seen_ids:
            all_candidates.append(c)
    
    with open(candidates_file, "w") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "total_candidates": len(all_candidates),
            "candidates": all_candidates,
        }, f, indent=2)
    
    print(f"\nFound {len(all_candidates)} unique candidate videos")
    print(f"Saved to: {candidates_file}")


def cmd_download(args):
    output_dir = Path(args.output_dir)
    candidates_file = output_dir / "candidates.json"
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    if not candidates_file.exists():
        print("No candidates.json found. Run 'search' first.")
        return
    
    with open(candidates_file) as f:
        data = json.load(f)
    
    candidates = data.get("candidates", [])
    
    to_download = [c for c in candidates if c.get("download_status") == "pending"]
    
    if args.max_downloads:
        to_download = to_download[:args.max_downloads]
    
    print(f"Downloading {len(to_download)} videos...")
    
    for candidate in tqdm(to_download, desc="Downloading"):
        vid = candidate["video_id"]
        url = candidate["url"]
        
        local_path = download_video(url, str(videos_dir), vid)
        
        if local_path:
            candidate["local_path"] = local_path
            candidate["download_status"] = "downloaded"
        else:
            candidate["download_status"] = "failed"
    
    with open(candidates_file, "w") as f:
        json.dump(data, f, indent=2)
    
    downloaded = sum(1 for c in candidates if c.get("download_status") == "downloaded")
    print(f"\nDownloaded: {downloaded} / {len(candidates)}")


def cmd_analyze(args):
    output_dir = Path(args.output_dir)
    candidates_file = output_dir / "candidates.json"
    
    if not candidates_file.exists():
        print("No candidates.json found. Run 'search' and 'download' first.")
        return
    
    try:
        from vlm.qwen_teacher import QwenTeacher, QwenTeacherConfig
    except ImportError:
        print("Could not import QwenTeacher. Make sure vlm/ is in your path.")
        return
    
    print(f"Loading VLM on {args.device}...")
    teacher = QwenTeacher(QwenTeacherConfig(
        device=args.device,
        prompt_mode="anticipation_simple",  # Fast but captures stages
        max_new_tokens=256,
        context_mode="none",
    ))
    
    with open(candidates_file) as f:
        data = json.load(f)
    
    candidates = data.get("candidates", [])
    
    to_analyze = [
        c for c in candidates 
        if c.get("download_status") == "downloaded" 
        and c.get("analysis_status") == "pending"
        and c.get("local_path")
    ]
    
    if args.max_analyze:
        to_analyze = to_analyze[:args.max_analyze]
    
    print(f"Analyzing {len(to_analyze)} videos...")
    
    for candidate in to_analyze:
        vid = candidate["video_id"]
        video_path = candidate["local_path"]
        
        print(f"\nAnalyzing: {vid}")
        print(f"  Title: {candidate.get('title', 'Unknown')[:60]}...")
        
        if not os.path.exists(video_path):
            print(f"  Video file not found: {video_path}")
            candidate["analysis_status"] = "failed"
            continue
        
        try:
            result = analyze_video_with_vlm(
                video_path,
                teacher,
                fps=args.fps,
                verbose=args.verbose,
            )
            
            if "error" in result:
                print(f"  Error: {result['error']}")
                candidate["analysis_status"] = "failed"
                candidate["notes"] = result["error"]
            else:
                candidate["analysis_status"] = "analyzed"
                candidate["has_violence"] = result.get("has_violence", False)
                candidate["violence_start_second"] = result.get("violence_start")
                candidate["escalation_start_second"] = result.get("escalation_start")
                candidate["anticipation_window_seconds"] = result.get("anticipation_window")
                candidate["stage_timeline"] = result.get("stage_timeline")
                
                if result.get("has_violence"):
                    print(f"  Violence detected at: {result['violence_start']:.1f}s")
                    if result.get("anticipation_window"):
                        print(f"  Anticipation window: {result['anticipation_window']:.1f}s")
                else:
                    print(f"  No violence detected")
                    
        except Exception as e:
            print(f"  Analysis failed: {e}")
            candidate["analysis_status"] = "failed"
            candidate["notes"] = str(e)
    
    with open(candidates_file, "w") as f:
        json.dump(data, f, indent=2)
    
    analyzed = sum(1 for c in candidates if c.get("analysis_status") == "analyzed")
    with_violence = sum(1 for c in candidates if c.get("has_violence"))
    with_anticipation = sum(
        1 for c in candidates 
        if c.get("anticipation_window_seconds") and c["anticipation_window_seconds"] >= 10
    )
    
    print(f"\n=== Summary ===")
    print(f"Analyzed: {analyzed}")
    print(f"With violence: {with_violence}")
    print(f"With 10+ second anticipation: {with_anticipation}")


def cmd_filter(args):
    output_dir = Path(args.output_dir)
    candidates_file = output_dir / "candidates.json"
    
    if not candidates_file.exists():
        print("No candidates.json found.")
        return
    
    with open(candidates_file) as f:
        data = json.load(f)
    
    candidates = data.get("candidates", [])
    
    good_candidates = []
    
    for c in candidates:
        if c.get("analysis_status") != "analyzed":
            continue
        if not c.get("has_violence"):
            continue
        
        anticipation = c.get("anticipation_window_seconds")
        if anticipation is None or anticipation < args.min_anticipation:
            continue
        
        # Heuristic: a longer lead-up (capped at 60 s) and a calm -> tension -> escalation progression score higher.
        quality = 0.0
        
        quality += min(anticipation / 60.0, 1.0) * 0.4
        
        timeline = c.get("stage_timeline", [])
        stages = [t.get("stage") for t in timeline]
        
        if "calm" in stages and "tension" in stages:
            quality += 0.2
        if "escalation" in stages:
            quality += 0.2
        
        if timeline and timeline[0].get("stage") == "violence":
            quality -= 0.3
        
        c["quality_score"] = max(0, min(1, quality))
        good_candidates.append(c)
    
    good_candidates.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    
    print(f"\n=== Good Anticipation Candidates ===")
    print(f"Found {len(good_candidates)} videos with {args.min_anticipation}+ second anticipation\n")
    
    for i, c in enumerate(good_candidates[:20]):
        print(f"{i+1}. {c['video_id']}")
        print(f"   Title: {c.get('title', 'Unknown')[:50]}...")
        print(f"   Anticipation: {c['anticipation_window_seconds']:.1f}s")
        print(f"   Quality: {c['quality_score']:.2f}")
        print()
    
    filtered_file = output_dir / "filtered_candidates.json"
    with open(filtered_file, "w") as f:
        json.dump({
            "min_anticipation": args.min_anticipation,
            "total": len(good_candidates),
            "candidates": good_candidates,
        }, f, indent=2)
    
    print(f"Saved to: {filtered_file}")


def cmd_export(args):
    output_dir = Path(args.output_dir)
    source_dir = Path(args.source_dir)
    filtered_file = source_dir / "filtered_candidates.json"
    
    if not filtered_file.exists():
        print("No filtered_candidates.json found. Run 'filter' first.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(filtered_file) as f:
        data = json.load(f)
    
    candidates = data.get("candidates", [])
    
    annotations = []
    
    for c in candidates:
        if not c.get("local_path") or not os.path.exists(c["local_path"]):
            continue
        
        src_path = c["local_path"]
        dst_path = output_dir / "videos" / f"{c['video_id']}.mp4"
        dst_path.parent.mkdir(exist_ok=True)
        
        if not dst_path.exists():
            import shutil
            shutil.copy2(src_path, dst_path)
        
        annotations.append({
            "video_id": c["video_id"],
            "video_path": str(dst_path.relative_to(output_dir)),
            "source": c.get("source", "unknown"),
            "title": c.get("title", ""),
            "duration_seconds": c.get("duration_seconds", 0),
            "violence_start_second": c.get("violence_start_second"),
            "escalation_start_second": c.get("escalation_start_second"),
            "anticipation_window_seconds": c.get("anticipation_window_seconds"),
            "quality_score": c.get("quality_score", 0),
        })
    
    ann_file = output_dir / "annotations.json"
    with open(ann_file, "w") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "total_videos": len(annotations),
            "annotations": annotations,
        }, f, indent=2)
    
    csv_file = output_dir / "annotations.csv"
    if annotations:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=annotations[0].keys())
            writer.writeheader()
            writer.writerows(annotations)
    
    print(f"\n=== Dataset Exported ===")
    print(f"Videos: {len(annotations)}")
    print(f"Output: {output_dir}")
    print(f"Annotations: {ann_file}")


def cmd_status(args):
    output_dir = Path(args.output_dir)
    candidates_file = output_dir / "candidates.json"
    
    if not candidates_file.exists():
        print("No candidates.json found. Run 'search' first.")
        return
    
    with open(candidates_file) as f:
        data = json.load(f)
    
    candidates = data.get("candidates", [])
    
    download_status = {}
    analysis_status = {}
    anticipation_counts = {
        "0-10s": 0,
        "10-30s": 0,
        "30-60s": 0,
        "60+s": 0,
    }
    
    for c in candidates:
        ds = c.get("download_status", "pending")
        download_status[ds] = download_status.get(ds, 0) + 1
        
        as_ = c.get("analysis_status", "pending")
        analysis_status[as_] = analysis_status.get(as_, 0) + 1
        
        ant = c.get("anticipation_window_seconds")
        if ant is not None:
            if ant < 10:
                anticipation_counts["0-10s"] += 1
            elif ant < 30:
                anticipation_counts["10-30s"] += 1
            elif ant < 60:
                anticipation_counts["30-60s"] += 1
            else:
                anticipation_counts["60+s"] += 1
    
    print(f"\n=== Pipeline Status ===")
    print(f"Total candidates: {len(candidates)}")
    print(f"\nDownload status:")
    for status, count in download_status.items():
        print(f"  {status}: {count}")
    print(f"\nAnalysis status:")
    for status, count in analysis_status.items():
        print(f"  {status}: {count}")
    print(f"\nAnticipation windows:")
    for window, count in anticipation_counts.items():
        print(f"  {window}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Curate violence anticipation dataset")
    parser.add_argument("--output_dir", type=str, default="data/anticipation_curation",
                       help="Output directory for curation pipeline")
    
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")
    
    search_parser = subparsers.add_parser("search", help="Search for candidate videos")
    search_parser.add_argument("--api_key", type=str, default=None,
                              help="YouTube Data API v3 key (or set YOUTUBE_API_KEY env var)")
    search_parser.add_argument("--max_per_query", type=int, default=10)
    search_parser.add_argument("--min_duration", type=int, default=30)
    search_parser.add_argument("--max_duration", type=int, default=600)
    
    download_parser = subparsers.add_parser("download", help="Download candidate videos")
    download_parser.add_argument("--max_downloads", type=int, default=None)
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze videos with VLM")
    analyze_parser.add_argument("--device", type=str, default="cuda:0")
    analyze_parser.add_argument("--fps", type=float, default=1.0,
                               help="Frames per second to analyze")
    analyze_parser.add_argument("--max_analyze", type=int, default=None)
    analyze_parser.add_argument("--verbose", action="store_true")
    
    filter_parser = subparsers.add_parser("filter", help="Filter for good anticipation candidates")
    filter_parser.add_argument("--min_anticipation", type=float, default=10,
                              help="Minimum anticipation window in seconds")
    
    export_parser = subparsers.add_parser("export", help="Export final dataset")
    export_parser.add_argument("--source_dir", type=str, default="data/anticipation_curation")
    
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    
    args = parser.parse_args()
    
    if args.command == "search":
        cmd_search(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "filter":
        cmd_filter(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()