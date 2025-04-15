# Audio Meeting Transcriber & Insight Extractor

## Overview

This Python application automates the transcription of internal meeting audio recordings and extracts actionable insights using the OpenAI ChatGPT API. It streamlines documentation and facilitates follow-up actions by converting spoken content into structured, analyzable text.

## Features

- **Audio Transcription**: Utilizes OpenAI's Whisper model to convert audio files into text.
- **Text Preprocessing**: Cleans and prepares transcribed text for analysis.
- **Insight Extraction**: Employs ChatGPT to summarize key points, identify issues, and suggest innovative solutions.
- **Modular Design**: Organized codebase with clear separation of concerns for maintainability.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/audio-meeting-transcriber.git
   cd audio-meeting-transcriber
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

5. **Ensure `ffmpeg` is installed**:
   - For Ubuntu/Debian:
     ```bash
     sudo apt-get install ffmpeg
     ```
   - For macOS (using Homebrew):
     ```bash
     brew install ffmpeg
     ```

## Usage

Run the application and follow the on-screen prompts:

```bash
python main.py
```

You will be prompted to choose one of the following options:

1. **Transcribe Audio Files**: Converts audio files in the specified directory to text.
2. **Process Transcribed Texts**: Analyzes existing transcriptions to extract insights.
3. **Full Pipeline**: Performs both transcription and analysis in sequence.

Ensure that your audio files are placed in the designated input directory before running the application.

## Directory Structure

```
audio-meeting-transcriber/
├── raw_audio/             # Input audio files
├── transcriptions/        # Output from audio transcription
├── preprocessed/          # Cleaned text ready for analysis
├── output/                # Final insights and summaries
├── utils/                 # Helper modules and classes
├── main.py                # Entry point of the application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## License

This project is licensed under the [MIT License](LICENSE).
