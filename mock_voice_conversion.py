import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import sounddevice as sd
import threading
import queue
import scipy.ndimage
import time

class VADMFCCDemo:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 22050
        self.frame_length = 2048  # ~93 ms at 22050 Hz
        self.hop_length = 512     # ~23 ms at 22050 Hz
        self.recording = False
        self.audio_queue = queue.Queue()
        self.buffer = np.zeros(0)
        self.vad_threshold = 0.4  # Default threshold for VAD
        self.is_voice = False
        
        # Create figure and subplots with more space
        plt.rcParams.update({'font.size': 10})  # Smaller font size
        self.fig, self.axes = plt.subplots(4, 1, figsize=(12, 12), 
                                          gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5]})
        self.fig.subplots_adjust(hspace=0.5, right=0.85)  # More space between plots and for colorbar
        
        # Set up the plot elements
        self.setup_plots()
        
        # Add threshold slider
        self.ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
        self.slider = plt.Slider(
            self.ax_slider, 'VAD Threshold', 0.0, 1.0, valinit=self.vad_threshold
        )
        self.slider.on_changed(self.update_threshold)
        
        # Add buttons
        self.ax_button_record = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.button_record = plt.Button(self.ax_button_record, 'Record')
        self.button_record.on_clicked(self.toggle_recording)
        
        # Title
        self.fig.suptitle("Voice Activity Detection with MFCC Demo\nStatus: Not Recording | Voice: Not Detected", 
                         fontsize=16)

    def setup_plots(self):
        # Waveform plot
        self.axes[0].set_title("Audio Waveform")
        self.axes[0].set_ylabel("Amplitude")
        self.waveform_line, = self.axes[0].plot([], [])
        self.axes[0].grid(True)
        self.axes[0].set_ylim(-1, 1)  # Fixed amplitude range
        # Remove x-ticks
        self.axes[0].set_xticks([])
        
        # Spectrogram plot - empty at initialization
        self.axes[1].set_title("Spectrogram")
        self.axes[1].set_ylabel("Frequency")
        dummy_data = np.zeros((128, 10))
        self.spectrogram_img = self.axes[1].imshow(
            dummy_data, aspect='auto', origin='lower', cmap='viridis'
        )
        # Simplified ticks
        self.axes[1].set_yticks([0, 64, 127])
        self.axes[1].set_yticklabels(['0', '4kHz', '8kHz'])
        self.axes[1].set_xticks([])
        # Colorbar with simpler labels - store reference
        self.cbar1 = plt.colorbar(self.spectrogram_img, ax=self.axes[1], pad=0.01)
        self.cbar1.set_label('Energy')
        
        # Mel spectrogram plot - empty at initialization
        self.axes[2].set_title("Mel Spectrogram")
        self.axes[2].set_ylabel("Mel Bins")
        self.mel_img = self.axes[2].imshow(
            dummy_data, aspect='auto', origin='lower', cmap='viridis'
        )
        # Simplified ticks
        self.axes[2].set_yticks([0, 64, 127])
        self.axes[2].set_yticklabels(['Low', 'Mid', 'High'])
        self.axes[2].set_xticks([])
        # Colorbar with simpler labels - store reference
        self.cbar2 = plt.colorbar(self.mel_img, ax=self.axes[2], pad=0.01)
        self.cbar2.set_label('Energy')
        
        # MFCC plot - empty at initialization
        self.axes[3].set_title("MFCCs")
        self.axes[3].set_ylabel("MFCC Coefficients")
        self.axes[3].set_xlabel("Time")
        dummy_mfcc = np.zeros((20, 10))
        self.mfcc_img = self.axes[3].imshow(
            dummy_mfcc, aspect='auto', origin='lower', cmap='viridis'
        )
        # Simplified ticks
        self.axes[3].set_yticks([0, 9, 19])
        self.axes[3].set_yticklabels(['1', '10', '20'])
        self.axes[3].set_xticks([])
        # Colorbar with simpler labels - store reference
        self.cbar3 = plt.colorbar(self.mfcc_img, ax=self.axes[3], pad=0.01)
        self.cbar3.set_label('Value')

    def update_threshold(self, val):
        self.vad_threshold = val

    def toggle_recording(self, event):
        if not self.recording:
            self.recording = True
            self.button_record.label.set_text('Stop')
            # Update title
            self.fig.suptitle("Voice Activity Detection with MFCC Demo\nStatus: Recording | Voice: Not Detected", 
                             fontsize=16)
            
            # Start recording in a separate thread
            threading.Thread(target=self.record_audio, daemon=True).start()
        else:
            self.recording = False
            self.button_record.label.set_text('Record')
            # Update title
            self.fig.suptitle("Voice Activity Detection with MFCC Demo\nStatus: Not Recording | Voice: Not Detected", 
                             fontsize=16)
    
    def record_audio(self):
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            # Convert to mono by averaging channels if stereo
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()
            
            # Add to queue
            self.audio_queue.put(audio_data)
        
        try:
            with sd.InputStream(callback=callback, channels=1, samplerate=self.sample_rate, 
                              blocksize=self.frame_length):
                while self.recording:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error recording audio: {e}")
            self.recording = False
    
    def update_plot(self, frame):
        # Get latest audio data from queue
        new_audio = []
        while not self.audio_queue.empty():
            try:
                new_audio.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        
        if new_audio:
            # Add new audio to buffer
            new_data = np.concatenate(new_audio)
            self.buffer = np.concatenate([self.buffer, new_data])
            
            # Keep only the last ~5 seconds
            max_buffer_size = 5 * self.sample_rate
            if len(self.buffer) > max_buffer_size:
                self.buffer = self.buffer[-max_buffer_size:]
            
            # Update waveform plot
            self.axes[0].clear()
            self.axes[0].plot(self.buffer)
            self.axes[0].set_ylabel("Amplitude")
            self.axes[0].set_ylim(-1, 1)  # Re-apply fixed amplitude range
            self.axes[0].grid(True)
            self.axes[0].set_xticks([])
            
            # Only process if we have enough data
            if len(self.buffer) >= self.frame_length:
                # Compute spectrogram
                D = librosa.stft(self.buffer, n_fft=self.frame_length, hop_length=self.hop_length)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                
                # Compute mel spectrogram
                S_mel = librosa.feature.melspectrogram(y=self.buffer, sr=self.sample_rate, 
                                                     n_fft=self.frame_length, hop_length=self.hop_length)
                S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
                
                # Compute MFCCs
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S_mel), n_mfcc=20)
                
                # Update spectrogram plot with minimal ticks
                self.axes[1].clear()
                self.axes[1].set_title("Spectrogram")
                self.axes[1].set_ylabel("Frequency")
                img1 = self.axes[1].imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
                # Update existing colorbar
                self.cbar1.mappable.set_array(S_db)
                self.cbar1.mappable.set_clim(vmin=np.min(S_db), vmax=np.max(S_db))
                # Re-apply simplified ticks
                self.axes[1].set_yticks([0, 64, 127])
                self.axes[1].set_yticklabels(['0', '4kHz', '8kHz'])
                self.axes[1].set_xticks([])
                
                # Update mel spectrogram plot with minimal ticks
                self.axes[2].clear()
                self.axes[2].set_title("Mel Spectrogram")
                self.axes[2].set_ylabel("Mel Bins")
                img2 = self.axes[2].imshow(S_mel_db, aspect='auto', origin='lower', cmap='viridis')
                # Update existing colorbar
                self.cbar2.mappable.set_array(S_mel_db)
                self.cbar2.mappable.set_clim(vmin=np.min(S_mel_db), vmax=np.max(S_mel_db))
                # Re-apply simplified ticks
                self.axes[2].set_yticks([0, 64, 127])
                self.axes[2].set_yticklabels(['Low', 'Mid', 'High'])
                self.axes[2].set_xticks([])
                
                # Update MFCC plot with minimal ticks
                self.axes[3].clear()
                self.axes[3].set_title("MFCCs")
                self.axes[3].set_ylabel("MFCC Coefficients")
                self.axes[3].set_xlabel("Time")
                img3 = self.axes[3].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
                # Update existing colorbar
                self.cbar3.mappable.set_array(mfcc)
                self.cbar3.mappable.set_clim(vmin=np.min(mfcc), vmax=np.max(mfcc))
                # Re-apply simplified ticks
                self.axes[3].set_yticks([0, 9, 19])
                self.axes[3].set_yticklabels(['1', '10', '20'])
                self.axes[3].set_xticks([])
                
                # Perform VAD decision
                # Using a simple energy-based approach with the first MFCC coefficient
                mfcc_energy = mfcc[0]
                # Normalize energy to 0-1 range
                if len(mfcc_energy) > 0 and np.max(mfcc_energy) > np.min(mfcc_energy):
                    mfcc_energy = (mfcc_energy - np.min(mfcc_energy)) / (np.max(mfcc_energy) - np.min(mfcc_energy))
                    # Smooth the energy curve
                    mfcc_energy = scipy.ndimage.gaussian_filter1d(mfcc_energy, sigma=2)
                    
                    # Make VAD decision based on the latest frame
                    if len(mfcc_energy) > 0:
                        recent_energy = mfcc_energy[-1]
                        self.is_voice = recent_energy > self.vad_threshold
                        
                        # Update title with VAD decision
                        if self.recording:
                            status = "Recording"
                        else:
                            status = "Not Recording"
                            
                        if self.is_voice:
                            voice = "VOICE DETECTED"
                            title_color = 'green'
                            self.axes[0].set_facecolor((0.9, 1.0, 0.9))  # Light green background
                        else:
                            voice = "NO VOICE"
                            title_color = 'red'
                            self.axes[0].set_facecolor((1.0, 0.9, 0.9))  # Light red background
                        
                        self.fig.suptitle(f"Voice Activity Detection with MFCC Demo\nStatus: {status} | Voice: {voice}", 
                                         fontsize=16, color=title_color)
                        
                        # Add decibel level information to waveform plot
                        if len(self.buffer) > 0:
                            rms = np.sqrt(np.mean(self.buffer**2))
                            db = 20 * np.log10(rms + 1e-9)  # Add small constant to avoid log(0)
                            self.axes[0].set_title(f"Audio Waveform - Level: {db:.1f} dB")
                        else:
                            self.axes[0].set_title("Audio Waveform")
            else:
                self.axes[0].set_title("Audio Waveform")
                self.axes[1].set_title("Spectrogram")
                self.axes[2].set_title("Mel Spectrogram")
                self.axes[3].set_title("MFCCs")

        # Return list of artists to redraw - minimal required for blitting=False
        # Returning axes contents might be safer if specific artists change
        return list(self.fig.get_axes()) # Return all axes to be redrawn

    def run(self):
        # Set up the animation
        self.ani = FuncAnimation(self.fig, self.update_plot, blit=False, interval=100)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for slider at bottom
        plt.show()

if __name__ == "__main__":
    demo = VADMFCCDemo()
    demo.run()