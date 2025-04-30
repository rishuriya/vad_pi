import sounddevice as sd
import numpy as np
import librosa
import joblib
import time
import queue
import threading
import torch
import csv
import os
from pyannote.audio import Model, Audio
from pyannote.core import Segment, Timeline
from pyannote.audio.pipelines import VoiceActivityDetection
# --- Configuration ---
CHUNK_DURATION = 15  # seconds
SAMPLE_RATE = 16000  # Target sample rate (adjust if necessary)
N_MFCC = 16          # Number of MFCC features (must match training)
N_FFT = 2048         # FFT window size for MFCC (adjust if necessary)
HOP_LENGTH = 512     # Hop length for MFCC (adjust if necessary)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available for Pyannote

# Paths to your trained model and scaler
# IMPORTANT: Update these paths to the actual location on the Raspberry Pi
SVM_MODEL_PATH = "/home/pi/vad_svm_from_csv_filename_labeled.joblib"
SCALER_PATH = "/home/pi/scaler_svm_from_csv_filename_labeled.joblib"

# CSV log file for mismatches
# IMPORTANT: Update this path if you want the log file elsewhere
MISMATCH_LOG_FILE = "/home/pi/rasperrypi/mismatch_log.csv"

# Pyannote model
PYANNOTE_MODEL_ID = "pyannote/segmentation-3.0"

# --- Environment Variables ---
HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Hugging Face token not found. Please set the HUGGING_FACE_TOKEN environment variable.")

# --- Global Variables ---
audio_queue = queue.Queue()
processing_active = True

# --- Load Models ---
print("Loading models...")
try:
    # Load SVM model and scaler
    svm_model = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded SVM model from: {SVM_MODEL_PATH}")
    print(f"Loaded Scaler from: {SCALER_PATH}")

    # Verify expected features (optional but recommended)
    if hasattr(svm_model, 'n_features_in_') and svm_model.n_features_in_ != N_MFCC:
         print(f" !!! WARNING: Loaded SVM model expects {svm_model.n_features_in_} features, but script is configured for {N_MFCC}. Check N_MFCC! ")
    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != N_MFCC:
         print(f" !!! WARNING: Loaded Scaler expects {scaler.n_features_in_} features, but script is configured for {N_MFCC}. Check N_MFCC! ")

    # Load Pyannote base model
    print(f"Loading base model: {PYANNOTE_MODEL_ID}")
    model = Model.from_pretrained(
        PYANNOTE_MODEL_ID,
        use_auth_token=HF_TOKEN
    )

    # Wrap model in VoiceActivityDetection pipeline
    print("Initializing VoiceActivityDetection pipeline...")
    vad_pipeline = VoiceActivityDetection(segmentation=model)

    # Set hyperparameters
    HYPER_PARAMETERS = {
        "min_duration_on": 0.5, # Minimum duration for a speech segment
        "min_duration_off": 0.5 # Minimum duration for a silence/noise gap between speech
    }
    vad_pipeline.instantiate(HYPER_PARAMETERS)
    vad_pipeline.to(torch.device(DEVICE))
    print(f"Loaded VAD Pipeline based on {PYANNOTE_MODEL_ID} on {DEVICE} with min_duration=0.5s")

    # Initialize Pyannote Audio utility (used for processing chunks)
    pyannote_audio = Audio(sample_rate=SAMPLE_RATE, mono=True)

except FileNotFoundError as e:
    print(f"Error: Model or scaler file not found. {e}")
    print("Please ensure the paths are correct.")
    exit(1)
except Exception as e:
    print(f"Error loading models: {e}")
    # Add more specific error info if possible
    import traceback
    traceback.print_exc()
    exit(1)

# --- Feature Calculation ---
def calculate_mfcc_features(audio_segment_np, sample_rate):
    """Calculates 16 MFCC features for a given audio segment."""
    # Ensure input is a flat, 1D numpy array
    audio_segment_np = audio_segment_np.flatten()

    if len(audio_segment_np) < N_FFT: # Pad if segment is too short for FFT
        pad_width = N_FFT - len(audio_segment_np)
        audio_segment_np = np.pad(audio_segment_np, (0, pad_width), mode='constant')

    if len(audio_segment_np) == 0:
        return np.zeros(N_MFCC) # Return zeros if segment is empty after potential padding issues

    try:
        mfccs = librosa.feature.mfcc(y=audio_segment_np,
                                     sr=sample_rate,
                                     n_mfcc=N_MFCC,
                                     n_fft=N_FFT,
                                     hop_length=HOP_LENGTH)
        # Aggregate MFCCs over time (using mean)
        if mfccs.shape[1] > 0:
            mean_mfccs = np.mean(mfccs, axis=1)
        else:
            mean_mfccs = np.zeros(N_MFCC) # Handle case where mfccs calculation results in empty array

        # Ensure the output is always the correct shape
        if mean_mfccs.shape[0] != N_MFCC:
             print(f"Warning: MFCC calculation resulted in unexpected shape {mean_mfccs.shape}. Returning zeros.")
             return np.zeros(N_MFCC)

        return mean_mfccs
    except Exception as e:
        print(f"Error calculating MFCCs: {e}")
        return np.zeros(N_MFCC) # Return zeros on error


# --- Audio Processing Thread ---
def process_audio():
    """Continuously processes audio chunks from the queue."""
    global processing_active
    print("Processing thread started.")
    while processing_active or not audio_queue.empty():
        try:
            # Get chunk data (NumPy array and original sample rate)
            audio_chunk_np, recording_time = audio_queue.get(timeout=1) # Wait 1 sec
            chunk_start_str = time.strftime("%H:%M:%S", time.localtime(recording_time))
            print(f"\nProcessing chunk recorded at {chunk_start_str}...")

            # --- Pyannote Segmentation ---
            try:
                # Create a Pyannote Audio object from the NumPy array
                # Pyannote expects shape (num_channels=1, num_samples)
                # Ensure tensor is moved to the same device as the pipeline
                # audio_chunk_np has shape (n_frames, 1) from sounddevice
                # Pyannote expects (1, n_frames)
                audio_tensor = torch.from_numpy(audio_chunk_np.T).float().to(DEVICE)
                pyannote_input = {"waveform": audio_tensor, "sample_rate": SAMPLE_RATE}

                # Call the VAD pipeline
                annotation = vad_pipeline(pyannote_input)

                # Extract timelines from the Annotation object
                # VAD pipeline typically labels speech segments as "SPEECH"
                speech_timeline = annotation.label_timeline("SPEECH").support()
                full_chunk_segment = Segment(0, CHUNK_DURATION)
                noise_timeline = Timeline([full_chunk_segment]).gaps(speech_timeline)

                print(f"  Pyannote found {len(speech_timeline)} speech and {len(noise_timeline)} noise segments.")

            except Exception as e:
                 print(f"  Error during Pyannote segmentation: {e}")
                 continue # Skip to next chunk if segmentation fails

            # --- Process Each Segment ---
            all_segments = list(speech_timeline) + list(noise_timeline)
            segment_labels = ([1] * len(speech_timeline)) + ([0] * len(noise_timeline)) # 1=Speech, 0=Noise (from Pyannote)

            for segment, pyannote_label in zip(all_segments, segment_labels):
                seg_start, seg_end = segment.start, segment.end
                if seg_end <= seg_start: continue # Skip zero-duration segments

                try:
                    # Extract the audio segment from the original chunk
                    start_sample = int(seg_start * SAMPLE_RATE)
                    end_sample = int(seg_end * SAMPLE_RATE)
                    segment_audio_np = audio_chunk_np[start_sample:end_sample]

                    if len(segment_audio_np) == 0: continue # Skip empty segments

                    # Calculate MFCC features
                    features = calculate_mfcc_features(segment_audio_np.astype(np.float32), SAMPLE_RATE)

                    if features is None or features.shape[0] != N_MFCC:
                         print(f"   Skipping segment [{seg_start:.2f}-{seg_end:.2f}s]: Feature calculation failed or wrong shape.")
                         continue

                    # Scale features
                    features_reshaped = features.reshape(1, -1)
                    features_scaled = scaler.transform(features_reshaped)

                    # Predict using SVM model
                    svm_prediction = svm_model.predict(features_scaled)[0]
                    svm_label = "Speech" if svm_prediction == 1 else "Noise"
                    pyannote_label_str = "Speech" if pyannote_label == 1 else "Noise"

                    print(f"   Segment [{seg_start:6.2f}-{seg_end:6.2f}s]: Pyannote='{pyannote_label_str}', Custom SVM='{svm_label}'")

                    # Log mismatch if labels differ
                    if svm_label != pyannote_label_str:
                        # Format MFCC features as a string for CSV
                        mfcc_string = '[' + ','.join(map(str, features)) + ']'
                        log_row = [
                            chunk_start_str,    # Timestamp of the chunk start
                            f"{seg_start:.3f}",     # Segment start time
                            f"{seg_end:.3f}",       # Segment end time
                            pyannote_label_str, # Pyannote label
                            svm_label,          # SVM label
                            mfcc_string         # MFCC features
                        ]
                        try:
                            with open(MISMATCH_LOG_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(log_row)
                            print(f"    -> Mismatch logged to {MISMATCH_LOG_FILE}")
                        except Exception as log_e:
                            print(f"    -> Error logging mismatch: {log_e}")

                except Exception as e:
                    print(f"   Error processing segment [{seg_start:.2f}-{seg_end:.2f}s]: {e}")

            audio_queue.task_done() # Mark task as complete

        except queue.Empty:
            # Queue is empty, wait or exit if recording stopped
            if not processing_active:
                break # Exit loop if recording has stopped and queue is empty
            else:
                continue # Continue waiting

        except Exception as e:
            print(f"Error in processing thread: {e}")
            audio_queue.task_done() # Ensure task_done is called even on error

    print("Processing thread finished.")


# --- Audio Recording Callback ---
def audio_callback(indata, frames, time_info, status):
    """This function is called by sounddevice for each new audio buffer."""
    if status:
        print(f"Recording status warning: {status}", flush=True)
    # Add the new data to the queue
    # We put a copy to avoid issues with the buffer being overwritten
    audio_queue.put((indata.copy(), time_info.inputBufferAdcTime))


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Initialize mismatch log file if it doesn't exist
        if not os.path.exists(MISMATCH_LOG_FILE):
            print(f"Creating mismatch log file: {MISMATCH_LOG_FILE}")
            with open(MISMATCH_LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ChunkTimestamp", "SegmentStart", "SegmentEnd", "PyannoteLabel", "SVMLabel", "MFCC_Features"])
        else:
            print(f"Appending to existing mismatch log file: {MISMATCH_LOG_FILE}")

        # Check default input device and sample rate
        device_info = sd.query_devices(kind='input')
        default_sr = int(device_info['default_samplerate'])
        print(f"Default input device: {device_info['name']}")
        print(f"Default sample rate: {default_sr} Hz")
        if default_sr != SAMPLE_RATE:
             print(f"Warning: Default sample rate ({default_sr}Hz) differs from target ({SAMPLE_RATE}Hz). Audio will be resampled by sounddevice.")

        # Start the processing thread
        processor_thread = threading.Thread(target=process_audio, daemon=True)
        processor_thread.start()

        # Calculate buffer size for the desired chunk duration
        blocksize = int(SAMPLE_RATE * CHUNK_DURATION)

        print(f"\nStarting {CHUNK_DURATION}s chunk recording (Sample Rate: {SAMPLE_RATE} Hz)... Press Ctrl+C to stop.")
        # Start recording stream
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=1,         # Mono
                            dtype='float32',    # Data type
                            blocksize=blocksize, # Process in CHUNK_DURATION blocks
                            callback=audio_callback):
            while True:
                time.sleep(0.1) # Keep main thread alive

    except KeyboardInterrupt:
        print("\nStopping recording...")
        processing_active = False # Signal processing thread to stop
        # Wait for the processing thread to finish remaining items
        print("Waiting for processing thread to finish...")
        audio_queue.join() # Wait for all queued items to be processed
        processor_thread.join(timeout=5) # Wait for thread itself to finish
        print("Recording stopped.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        processing_active = False
        # Attempt graceful shutdown
        if 'processor_thread' in locals() and processor_thread.is_alive():
             audio_queue.join()
             processor_thread.join(timeout=5)

    print("Script finished.") 