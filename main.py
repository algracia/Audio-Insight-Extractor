import os
import shutil
import nltk
from utils.audioTranscriber import AudioTranscriber
from utils.chatGPT_api import ChatGPTapi


def setup_nltk_resources(nltk_data_path):
    """Ensure required NLTK resources are available."""
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    for resource in ["tokenizers/punkt", "corpora/stopwords"]:
        try:
            nltk.data.find(resource)
            print(f"[INFO] The '{resource.split('/')[-1]}' resource is already available.")
        except LookupError:
            print(f"[INFO] Downloading '{resource.split('/')[-1]}' resource...")
            nltk.download(resource.split('/')[-1], download_dir=nltk_data_path)


def transcribe_audios(input_folder, output_folder):
    """Transcribe audio files from a given folder."""
    print("Starting audio transcription...")
    transcriber = AudioTranscriber(input_folder=input_folder, output_folder=output_folder, model_type="large", device="cpu")
    transcriber.process_all_audios()
    print("Audio transcription completed.")


def clean_folder(folder):
    """Delete all contents inside a folder."""
    if not os.path.exists(folder):
        print(f"[INFO] Folder {folder} does not exist. No cleanup needed.")
        return

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"[ERROR] Could not delete {file_path}: {e}")


def process_texts(output_folder, consolidated_output, preprocessed_folder):
    """Analyze transcribed text files using ChatGPT."""
    print("Starting text analysis with ChatGPT...")
    chatGPT = ChatGPTapi()

    os.makedirs(preprocessed_folder, exist_ok=True)
    print("[INFO] Cleaning the preprocessed folder...")
    clean_folder(preprocessed_folder)

    with open(consolidated_output, "w", encoding="utf-8") as f:
        f.write("Consolidated Responses from ChatGPT\n")
        f.write("=" * 50 + "\n\n")

    for file_name in os.listdir(output_folder):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(output_folder, file_name)
        print(f"Processing file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                audio_text = file.read()

            if not audio_text.strip():
                print(f"The file {file_name} is empty or contains no valid text.")
                continue

            preprocessed_text = chatGPT.preprocess_text(audio_text)

            preprocessed_path = os.path.join(preprocessed_folder, f"preprocessed_{file_name}")
            with open(preprocessed_path, "w", encoding="utf-8") as preprocessed_file:
                preprocessed_file.write(preprocessed_text)

            prompt = f"""
            You are an assistant specialized in text summarization and analysis.

            This transcript is a voice-over narration used in a demo video. The speaker is explaining the features of an app that transcribes audio recordings of internal meetings and then processes them using the ChatGPT API.

            Please analyze the transcript and:
            1. Extract the main points covered by the speaker.
            2. Highlight what makes the app useful or innovative.
            3. Suggest a professional summary that could be used for documentation or marketing purposes.
            4. Clarify any awkward or unclear phrasing due to speech-to-text transcription errors.

            Keep your response concise and well-structured.

            Text to analyze:
            {preprocessed_text}
            """

            summary_path = os.path.join(output_folder, f"summary_{file_name}")
            response = chatGPT.analyze_in_parts("Meeting transcript analysis", preprocessed_text, prompt, summary_path)

            if not response or "Parece que no has proporcionado" in response:
                print(f"Could not process file {file_name}. Please check the text format.")
                continue

            with open(consolidated_output, "a", encoding="utf-8") as f:
                f.write(f"Processed file: {file_name}\n")
                f.write(f"ChatGPT Response:\n{response}\n")
                f.write("=" * 50 + "\n\n")

            print(f"Processed and analyzed: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print(f"Analysis completed. Consolidated responses saved to: {consolidated_output}")


def main():
    base_dir = "/home/algraciam/Desktop/Procesamiento de Audios/AppAudios"
    input_folder = os.path.join(base_dir, "raw audios")
    output_folder = os.path.join(base_dir, "transcriptions")
    preprocessed_folder = os.path.join(base_dir, "preprocessed")
    consolidated_output = os.path.join(base_dir, "consolidated output", "respuesta_consolidada.txt")

    setup_nltk_resources(os.path.join(base_dir, "ConvertirAudioEnCodigo", "nltk_data"))

    print("Select an option:")
    print("1. Only transcribe audios.")
    print("2. Only process texts with ChatGPT.")
    print("3. Do both (transcribe and process texts).")
    option = input("Enter the number of your option: ")

    if option == "1":
        transcribe_audios(input_folder, output_folder)
    elif option == "2":
        process_texts(output_folder, consolidated_output, preprocessed_folder)
    elif option == "3":
        transcribe_audios(input_folder, output_folder)
        process_texts(output_folder, consolidated_output, preprocessed_folder)
    else:
        print("Invalid option. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()