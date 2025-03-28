"""Utilities for WEBVTT files"""

import re

import re

def merge_webvtt_to_list(webvtt_str: str, merge_seconds: int) -> list[dict[str, str]]:
    """Merge WEBVTT subtitles into a list of dictionaries with merged text and initial time."""
    
    lines = [line.strip() for line in webvtt_str.split("\n") if line.strip()]
    if not lines or lines[0] != "WEBVTT":
        raise ValueError("Invalid WEBVTT format")
    
    lines = lines[1:]  # Skip "WEBVTT" header
    time_regex = r"(\d{2}):(\d{2}):(\d{2})\.\d{3}"

    result = []
    merged_text = ""
    initial_time = None
    cumulative_time = 0  # Tracks merged block time

    for i, line in enumerate(lines):
        if "-->" in line:
            times = re.findall(time_regex, line)
            start_time = int(times[0][0]) * 3600 + int(times[0][1]) * 60 + int(times[0][2])
            end_time = int(times[1][0]) * 3600 + int(times[1][1]) * 60 + int(times[1][2])
            
            if initial_time is None:  
                initial_time = start_time  # Set initial time for first block
            
            duration = end_time - initial_time
            cumulative_time += duration

        else:
            merged_text += line + " "

            if cumulative_time >= merge_seconds:
                result.append({
                    "text": merged_text.strip(),
                    "initial_time_in_seconds": initial_time,
                })
                merged_text = ""
                initial_time = end_time  # Reset start time for new block
                cumulative_time = 0  # Reset cumulative time

    # Add remaining text if any
    if merged_text.strip():
        result.append({
            "text": merged_text.strip(),
            "initial_time_in_seconds": initial_time,
        })

    return result
