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
import requests
import json
from datetime import datetime
import pathlib # Added for home directory
from pyannote.audio import Model, Audio
from pyannote.core import Segment, Timeline
from pyannote.audio.pipelines import VoiceActivityDetection

# --- Configuration ---
CHUNK_DURATION = 15  # seconds
# Default sample rate (will be updated with actual device rate)
default_sr = 16000  # Initial fallback value
N_MFCC = 16          # Number of MFCC features (must match training)
N_FFT = 2048         # FFT window size for MFCC (adjust if necessary)
HOP_LENGTH = 512     # Hop length for MFCC (adjust if necessary)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available for Pyannote
MIN_SEGMENT_DURATION = 0.1 # Minimum duration (seconds) for a segment to be processed
API_ENDPOINT = "https://noiser-u6n9.onrender.com/api/noise-data"
RETRY_INTERVAL = 60  # Time between retry attempts (seconds)
MAX_QUEUE_SIZE = 1000 # Maximum number of failed API requests to store

# Get user's home directory
HOME_DIR = pathlib.Path.home()

# --- Paths --- 
# Construct paths relative to the user's home directory
# IMPORTANT: Adjust the relative paths ('vad_project/...') if your files are in a different subdirectory
SVM_MODEL_PATH = HOME_DIR.joinpath("vad_project/vad_svm_from_csv_filename_labeled.joblib")
SCALER_PATH = HOME_DIR.joinpath("vad_project/scaler_svm_from_csv_filename_labeled.joblib")

# CSV log file for mismatches
# Create rasperrypi directory in home if it doesn't exist
LOG_DIR = HOME_DIR.joinpath("vad_project/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
MISMATCH_LOG_FILE = LOG_DIR.joinpath("mismatch_log.csv")

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
    pyannote_audio = Audio(sample_rate=default_sr, mono=True)

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
    device_info = sd.query_devices(kind='input')
    default_sr = int(device_info['default_samplerate'])
    global processing_active
    
    # Access the logger
    global logger
    
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
                pyannote_input = {"waveform": audio_tensor, "sample_rate": default_sr}

                # Call the VAD pipeline
                annotation = vad_pipeline(pyannote_input)

                # --- DEBUG PRINTS ---
                print(f"  Raw Pyannote Annotation for chunk: {annotation}")
                # --- END DEBUG PRINTS ---

                # Extract timelines from the Annotation object
                # VAD pipeline typically labels speech segments as "SPEECH"
                speech_timeline = annotation.label_timeline("SPEECH").support()
                
                # --- Manually Calculate Noise Segments (Workaround for .gaps() issue) ---
                noise_segments = []
                last_end_time = 0.0
                # Sort speech segments just in case (should already be sorted)
                sorted_speech_segments = sorted(list(speech_timeline), key=lambda s: s.start)

                for speech_segment in sorted_speech_segments:
                    if speech_segment.start > last_end_time:
                        # Add the gap before this speech segment as noise
                        noise_segments.append(Segment(last_end_time, speech_segment.start))
                    last_end_time = speech_segment.end # Update the end time
                
                # Add the final gap after the last speech segment (if any)
                if last_end_time < CHUNK_DURATION:
                    noise_segments.append(Segment(last_end_time, CHUNK_DURATION))
                
                # Create the noise timeline from the calculated segments
                noise_timeline = Timeline(noise_segments)
                # --- End Manual Calculation ---

                # --- Filter Short Segments ---
                original_speech_count = len(speech_timeline)
                original_noise_count = len(noise_timeline)
                speech_timeline = Timeline([s for s in speech_timeline if s.duration >= MIN_SEGMENT_DURATION])
                noise_timeline = Timeline([s for s in noise_timeline if s.duration >= MIN_SEGMENT_DURATION])
                filtered_speech_count = len(speech_timeline)
                filtered_noise_count = len(noise_timeline)
                if original_speech_count != filtered_speech_count or original_noise_count != filtered_noise_count:
                    print(f"  Filtered out {original_speech_count - filtered_speech_count} short speech and {original_noise_count - filtered_noise_count} short noise segments (min dur: {MIN_SEGMENT_DURATION}s)")
                # --- End Filter Short Segments ---

                # --- DEBUG PRINTS (using filtered timelines) ---
                print(f"  Derived Speech Timeline segments (filtered): {len(speech_timeline)}")
                print(f"  Raw noise_timeline object (manual, filtered): {noise_timeline}")
                print(f"  Derived Noise Timeline segments (manual, filtered): {len(noise_timeline)}")
                # --- END DEBUG PRINTS ---

                print(f"  Pyannote found {len(speech_timeline)} speech and {len(noise_timeline)} noise segments (after filtering)." )

                # --- Process Noise Segments for API ---
                if len(noise_timeline) > 0:
                    print(f"Processing {len(noise_timeline)} noise segments for API submission")
                    
                    # Calculate average noise level across all noise segments
                    noise_levels = []
                    
                    # Collect noise data from all noise segments
                    for noise_segment in noise_timeline:
                        try:
                            seg_start, seg_end = noise_segment.start, noise_segment.end
                            start_sample = int(seg_start * default_sr)
                            end_sample = int(seg_end * default_sr)
                            
                            # Extract the audio segment
                            segment_audio_np = audio_chunk_np[start_sample:end_sample]
                            
                            if len(segment_audio_np) > 0:
                                # Calculate dB level
                                db_level = calculate_db_level(segment_audio_np.flatten())
                                noise_levels.append(db_level)
                                print(f"Noise segment [{seg_start:.2f}-{seg_end:.2f}s]: {db_level} dB")
                        except Exception as e:
                            print(f"Error processing noise segment for API: {e}")
                    
                    # If we found any valid noise levels, send the average to the API
                    if noise_levels:
                        avg_noise_level = sum(noise_levels) / len(noise_levels)
                        print(f"Average noise level: {avg_noise_level:.2f} dB")
                        
                        # Try to send to API directly first
                        success = send_to_api(avg_noise_level)
                        
                        # If direct send fails, queue it for later
                        if not success:
                            queue_noise_data(avg_noise_level)

            except Exception as e:
                 print(f"  Error during Pyannote segmentation: {e}")
                 continue # Skip to next chunk if segmentation fails

            # --- Process Each Segment (SVM vs Pyannote comparison) ---
            all_segments = list(speech_timeline) + list(noise_timeline)
            segment_labels = ([1] * len(speech_timeline)) + ([0] * len(noise_timeline)) # 1=Speech, 0=Noise (from Pyannote)

            for segment, pyannote_label in zip(all_segments, segment_labels):
                seg_start, seg_end = segment.start, segment.end
                if seg_end <= seg_start: continue # Skip zero-duration segments

                try:
                    # Extract the audio segment from the original chunk
                    start_sample = int(seg_start * default_sr)
                    end_sample = int(seg_end * default_sr)
                    segment_audio_np = audio_chunk_np[start_sample:end_sample]

                    if len(segment_audio_np) == 0: continue # Skip empty segments

                    # Calculate MFCC features
                    features = calculate_mfcc_features(segment_audio_np.astype(np.float32), default_sr)

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


# --- API Integration ---
# Queue for failed API requests
api_request_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
last_api_attempt_time = 0  # Track the last time we attempted an API call

# Function to get geolocation from IP
def get_location_from_ip():
    try:
        # Use a free IP geolocation API service
        response = requests.get('https://ipinfo.io/json')
        if response.status_code == 200:
            data = response.json()
            # Extract coordinates (format: "lat,lng")
            if 'loc' in data and data['loc']:
                lat, lng = data['loc'].split(',')
                return float(lat), float(lng)
            else:
                print("Warning: No location data in IP response")
        else:
            print(f"Warning: Failed to get IP location: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error getting location from IP: {e}")
    
    # Default fallback coordinates if lookup fails (0,0 - null island)
    return 0.0, 0.0

# Calculate dB level from audio segment
def calculate_db_level(audio_segment):
    """Calculate the dB level of an audio segment, ensure it's positive."""
    # Use RMS energy to calculate dB
    if len(audio_segment) == 0:
        return 0.0
    
    # Calculate RMS energy
    rms = np.sqrt(np.mean(np.square(audio_segment)))
    
    # Convert to dB (avoid log of zero)
    if rms > 0:
        db = 20 * np.log10(rms)
        # Convert to positive scale (typical environmental noise is 30-90 dB)
        # We'll use a reference where 0 dB RMS becomes 30 dB environmental
        positive_db = max(30 + db, 0)  # Ensure it's not negative
    else:
        positive_db = 0.0
    
    # Convert numpy type to Python native float
    return float(round(positive_db, 2))

# Send data to API endpoint
def send_to_api(noise_level, audio_type="ambient"):
    """Send noise data to the API endpoint."""
    try:
        # Get current timestamp in ISO format
        timestamp = datetime.now().isoformat()
        
        # Get location from IP
        latitude, longitude = get_location_from_ip()
        
        # Prepare payload - ensure all types match API expectations
        payload = {
            "latitude": str(latitude),  # Convert float to string
            "longitude": str(longitude),  # Convert float to string
            "noise_level": int(round(noise_level)),  # Convert float to integer
            "timestamp": timestamp,
            "location": "raspberry pi",
            "audio_type": audio_type
        }
        
        # Send POST request
        response = requests.post(API_ENDPOINT, json=payload, timeout=5)
        
        if response.status_code == 200:
            print(f"Successfully sent noise data to API: {noise_level} dB")
            return True
        else:
            print(f"Warning: API request failed with status code {response.status_code}")
            return False
    
    except requests.RequestException as e:
        print(f"Error: API request error: {e}")
        return False
    except Exception as e:
        print(f"Error: Error sending to API: {e}")
        return False

# Function to add failed request to queue
def queue_noise_data(noise_level, audio_type="ambient"):
    """Queue the noise data for later retry if immediate send fails."""
    try:
        # Get current timestamp in ISO format
        timestamp = datetime.now().isoformat()
        
        # Get location from IP (we'll cache this to avoid repeated lookups)
        if not hasattr(queue_noise_data, 'cached_location'):
            queue_noise_data.cached_location = get_location_from_ip()
        
        latitude, longitude = queue_noise_data.cached_location
        
        # Prepare data for queue - ensure all types match API expectations
        noise_data = {
            "latitude": str(latitude),  # Convert float to string
            "longitude": str(longitude),  # Convert float to string
            "noise_level": int(round(noise_level)),  # Convert float to integer
            "timestamp": timestamp,
            "location": "raspberry pi",
            "audio_type": audio_type
        }
        
        # Add to queue, remove oldest if full
        try:
            api_request_queue.put_nowait(noise_data)
            print(f"Queued noise data: {int(round(noise_level))} dB (queue size: {api_request_queue.qsize()})")
        except queue.Full:
            # Remove the oldest item and add the new one
            try:
                api_request_queue.get_nowait()
                api_request_queue.put_nowait(noise_data)
                print(f"Warning: Queue full, dropped oldest item to add new: {int(round(noise_level))} dB")
            except Exception as qe:
                print(f"Error managing queue: {qe}")
    
    except Exception as e:
        print(f"Error queueing noise data: {e}")

# Thread to process the API request queue
def process_api_queue():
    """Process the queue of failed API requests."""
    print("API queue processor thread started")
    
    while processing_active:
        try:
            # Check if we should attempt API communication based on time
            current_time = time.time()
            time_since_last_attempt = current_time - process_api_queue.last_attempt_time
            
            if time_since_last_attempt >= RETRY_INTERVAL:
                # Try to send a test ping first to see if API is responsive
                try:
                    test_response = requests.get(API_ENDPOINT.rsplit('/', 1)[0], timeout=2)
                    api_available = test_response.status_code < 500  # Any non-server error
                except:
                    api_available = False
                
                process_api_queue.last_attempt_time = current_time
                
                if api_available and not api_request_queue.empty():
                    print(f"API appears available, processing queue (size: {api_request_queue.qsize()})")
                    
                    # Process up to 10 items at once to avoid flooding
                    success_count = 0
                    failure_count = 0
                    max_batch = min(10, api_request_queue.qsize())
                    
                    for _ in range(max_batch):
                        if api_request_queue.empty():
                            break
                            
                        try:
                            # Get item but don't remove yet
                            noise_data = api_request_queue.queue[0]
                            
                            # Try to send
                            response = requests.post(API_ENDPOINT, json=noise_data, timeout=5)
                            
                            if response.status_code == 200:
                                # Successful, remove from queue
                                api_request_queue.get_nowait()
                                success_count += 1
                            else:
                                # API is up but rejected the request, log and remove
                                print(f"Warning: API rejected queued data: {response.status_code}")
                                api_request_queue.get_nowait()
                                failure_count += 1
                                
                        except requests.RequestException:
                            # API seems down again, break batch processing
                            print("Warning: API communication failed during queue processing")
                            failure_count += 1
                            break
                        except Exception as e:
                            # Other processing error, remove item to avoid queue blocking
                            print(f"Error processing queued item: {e}")
                            try:
                                api_request_queue.get_nowait()
                            except:
                                pass
                            failure_count += 1
                    
                    print(f"Queue processing results: {success_count} sent, {failure_count} failed")
            
            # Sleep before next check
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in API queue processor: {e}")
            time.sleep(5)  # Longer sleep on error
    
    print("API queue processor thread stopping")

# Initialize last attempt time
process_api_queue.last_attempt_time = 0

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
        # No need to compare with a constant SAMPLE_RATE anymore

        # Start the processing thread
        processor_thread = threading.Thread(target=process_audio, daemon=True)
        processor_thread.start()
        
        # Start the API queue processor thread
        api_queue_thread = threading.Thread(target=process_api_queue, daemon=True)
        api_queue_thread.start()
        print("API queue processor thread started")

        # Calculate buffer size for the desired chunk duration
        blocksize = int(default_sr * CHUNK_DURATION)

        print(f"\nStarting {CHUNK_DURATION}s chunk recording (Sample Rate: {default_sr} Hz)... Press Ctrl+C to stop.")
        # Start recording stream
        with sd.InputStream(samplerate=default_sr,
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