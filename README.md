# VAD Pi

A Voice Activity Detection (VAD) system for Raspberry Pi that automatically detects and processes audio segments, distinguishing between speech and ambient noise.

## Overview

This project sets up a real-time Voice Activity Detection system on a Raspberry Pi. It uses both a traditional SVM model and the Pyannote neural VAD pipeline to accurately detect speech vs. ambient noise. The system can run as a service on boot and automatically connects to WiFi.

## Features

- Real-time audio processing
- Dual VAD detection (SVM model + Pyannote)
- Automatic WiFi connection
- Systemd service integration for running on boot
- API integration for data submission
- GPU acceleration support (when available)

## Requirements

- Raspberry Pi (Tested on Raspberry Pi 3 and 4)
- Microphone (USB or built-in)
- Internet connection
- Hugging Face account with an access token

## Installation

### Option 1: One-Click Installation (Recommended)

Run this single command to download and install everything automatically:

```bash
sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/rishuriya/vad_pi/main/install_vad_pi.sh)"
```

This will:
1. Clone the repository
2. Make all scripts executable
3. Install dependencies and set up services
4. Configure WiFi and permissions
5. Guide you through the entire installation process

### Option 2: Manual Setup with Helper Script

If you've already cloned the repository:

```bash
sudo bash ./run_setup.sh
```

This will:
1. Install all required dependencies
2. Configure WiFi connection
3. Set up Python environment and install requirements
4. Configure your Hugging Face token
5. Set up systemd services for automatic startup
6. Guide you through the rest of the setup process

### Option 3: Fully Manual Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/rishuriya/vad_pi.git
   cd vad_pi
   ```

2. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y git python3 python3-venv network-manager wget portaudio19-dev
   ```

3. Create a virtual environment:
   ```bash
   python3 -m venv ~/venv_vad
   source ~/venv_vad/bin/activate
   pip install -r requirements_inference.txt
   ```

4. Set your Hugging Face token:
   ```bash
   export HUGGING_FACE_TOKEN="your_token_here"
   ```

5. Run the VAD inference script:
   ```bash
   python realtime_vad_inference.py
   ```

## Configuration

The main configuration parameters can be found at the top of the `realtime_vad_inference.py` file:

- `CHUNK_DURATION`: Duration in seconds for each audio processing chunk
- `N_MFCC`: Number of MFCC features
- `MIN_SEGMENT_DURATION`: Minimum duration for a speech/noise segment to be processed
- `API_ENDPOINT`: Endpoint for data submission

## Usage

Once installed and running, the system will:

1. Continuously monitor audio input
2. Classify segments as speech or ambient noise
3. Calculate noise levels for ambient noise segments
4. Send noise data to the configured API endpoint (if enabled)

To check if the service is running:

```bash
sudo systemctl status vad-inference.service
```

To view logs:

```bash
sudo journalctl -u vad-inference.service -f
```

## Troubleshooting

- **WiFi Connection Issues**: Check the status of the WiFi connection service with:
  ```bash
  sudo systemctl status wifi-connect.service
  ```

- **Audio Input Issues**: Make sure your microphone is properly connected and recognized:
  ```bash
  arecord -l
  ```

- **Service Not Starting**: Check for errors in the service logs:
  ```bash
  sudo journalctl -u vad-inference.service -e
  ```

- **Permission Issues**: If the service fails due to permissions, try manually setting permissions:
  ```bash
  sudo chmod +x ~/vad_project/realtime_vad_inference.py
  sudo systemctl restart vad-inference.service
  ```

## License

[MIT License](LICENSE)

## Acknowledgments

- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) for the neural VAD implementation
- [librosa](https://librosa.org/) for audio feature extraction 