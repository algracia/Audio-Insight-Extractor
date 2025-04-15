import os
import whisper


class AudioTranscriber:
    def __init__(self, input_folder: str, output_folder: str, model_type: str, device: str):
        """
        Initializes the AudioTranscriber class with the necessary configurations.

        Args:
            input_folder (str): Folder containing the audio files.
            output_folder (str): Folder where the transcriptions will be saved.
            model_type (str): Type of Whisper model to load.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_type = model_type
        self.device = device

        self.model = self._load_model()
        os.makedirs(self.output_folder, exist_ok=True)

    def _load_model(self):
        """
        Loads the Whisper model using the specified configuration.

        Returns:
            whisper.Whisper: Loaded Whisper model.
        """
        print(f"[INFO] Loading model '{self.model_type}' on {self.device}...")
        try:
            model = whisper.load_model(self.model_type, device=self.device)
            print(f"[INFO] Model loaded successfully.")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribes an audio file using the Whisper model.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            str: Transcribed text.
        """
        try:
            result = self.model.transcribe(file_path)
            print(f"[INFO] Transcription completed. Detected language: {result['language']}")
            return result["text"]
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to transcribe {file_path}: {e}")

    def process_all_audios(self):
        """
        Processes all supported audio files in the input folder.
        Saves each transcription as a .txt file in the output folder.
        """
        for file_name in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file_name)

            if file_name.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".aac")):
                try:
                    print(f"[INFO] Processing: {file_name}")
                    transcription = self.transcribe_audio(file_path)

                    output_file = os.path.join(self.output_folder, f"{os.path.splitext(file_name)[0]}.txt")
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(transcription)

                    print(f"[INFO] Saved transcription to: {output_file}")
                except Exception as e:
                    print(f"[ERROR] Error processing {file_name}: {e}")
            else:
                print(f"[WARNING] Unsupported file skipped: {file_name}")

        print("[INFO] Audio processing completed.")
