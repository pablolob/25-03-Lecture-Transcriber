# Lecture Transcriber 😊

A Python tool to transcribe lectures for LLM post-processing.

## Features
- Utilizes [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for efficient transcription.
- Supports `--regex` filtering to select specific files for transcription.
  - Example: python.exe ".\Lectures Directory" --regex  ".*24-11-15.*"


## Installation
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ``` 
## Performance

Depends on file size and format but for .mp4 of 2 hrs Lectures (around 250Mb) is averaging 5-6 minutes on a RTX 4060 GPU

Open to improvements : )
