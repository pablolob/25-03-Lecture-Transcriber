import argparse, time, re, os, subprocess 
import gc # Garbage Collector
import torch
from faster_whisper import WhisperModel

# Costants
FOLDER = r'.'
DESTINY_FOLDER = r'.\transcript'
FOLDER_AUDIO = r'.\audio'

# Timing decorator
def execution_time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ Tiempo de ejecución de '{func.__name__}': {end_time - start_time:.2f} segundos.")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        return result
    return wrapper

# Tried converting to audio first but not more eficient
def video2audio(file_path, destiny_folder, print_output=False):
    # Convertir el video a audio
    audio_file = os.path.join(destiny_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.wav")
    if os.path.exists(audio_file):
        if print_output:
            print(f"El archivo {audio_file} ya existe. Saltando conversión.")
        return audio_file
    
    if print_output:
        print(f"Convirtiendo {file_path} a audio...")
    command = [
        "ffmpeg", "-i", file_path,  # Archivo de entrada
        "-acodec", "pcm_s16le",      # Formato WAV sin compresión
        "-ar", "16000",              # Frecuencia de muestreo 16kHz (óptimo para Whisper)
        "-ac", "1",                  # Audio mono (Whisper no necesita estéreo)
        audio_file                   # Archivo de salida
    ]
    
    try:
        time_start = time.time()
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if print_output:
            print(f"✅ Conversión completada en {time.time() - time_start:.2f} segundos: {audio_file}")
    except subprocess.CalledProcessError as e:
        
        print(f"❌ Error en FFmpeg: {e}")
    return audio_file

# Transcribir
def transcribe_audio(file_path, model, beam_size: int = 3, print_output: bool = False) -> str:
    
    # Cargar el modelo WhisperModel "large"
    
    start_time = time.time()
    if print_output:
        print(f"Transcribiendo: {file_path}...")
    
    # Transcribir en tiempo real con parámetros optimizados
    transcription = []
    segments, _ = model.transcribe(file_path, beam_size=beam_size, word_timestamps=True, language="es")
    
    for segment in segments:
        transcription.append(str(segment.start)+segment.text)
    
    end_time = time.time()

    resultado = " ".join(transcription)
    
    if print_output:
        print(f"✅ Transcripción terminada en {end_time - start_time:.2f} segundos.")
    return resultado 

# @execution_time_decorator
def conversion_completa(file_path, destiny_folder, audio_destiny_folder, model, model_size: str = "large-v2", device: str = "cuda", beam_size: int = 2, print_output: bool = False, Bvideo2audio: bool = True) -> str:
    # Pre-conversion
    if Bvideo2audio:
        audio_file = video2audio(file_path, audio_destiny_folder, print_output = print_output)
    else:
        audio_file = file_path
    
    resultado = ""
    resultado = transcribe_audio(audio_file, model, model_size, device, beam_size, print_output= print_output)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    output_file = os.path.join(destiny_folder, f"{file_name}.txt")
    
    # Writing the transcription to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(resultado)

    print(f"Transcripción guardada en: {output_file}")

    del resultado  # Eliminar la variable de la memoria
    gc.collect()  # Limpiar la memoria
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  
        print("Memoria de la GPU liberada.")

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Split videos in a folder based on a regular expression.")
    parser.add_argument('--regex', type=str, default=".*", help="Regular expression to filter video files.")
    args = parser.parse_args()
    regex = args.regex
    

    model = WhisperModel("large-v2", device="cuda", compute_type="float16")
    
    # Main Loop
    for filename in os.listdir(FOLDER):
        try:    
            if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')) and re.match(regex, filename):
                # print(os.path.join(FOLDER, filename), os.path.join(DESTINY_FOLDER, f"{os.path.splitext(filename)[0]}.txt"))     
                file_path = os.path.join(FOLDER, filename)
                output_file = os.path.join(DESTINY_FOLDER, f"{os.path.splitext(filename)[0]}.txt")
                
                if os.path.exists(output_file):
                    print(f"El archivo {output_file} ya existe. Saltando transcripción.")
                    
                else:

                    conversion_completa(file_path, DESTINY_FOLDER, FOLDER_AUDIO, model, print_output=True, Bvideo2audio=False)
                    
                    time.sleep(4)
        except Exception as e:
            print(f"Error en el archivo {filename}: {e}")
    
    print("Proceso terminado. 😊")