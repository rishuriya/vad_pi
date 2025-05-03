#!/bin/bash

# --- Configuration ---
# !!! IMPORTANT: Set these variables before making the script available for download !!!
KNOWN_WIFI_SSID="Rishav's iPhone"
KNOWN_WIFI_PSK="wefp8772"

# This is the URL of the *actual project repository* to be cloned
GIT_REPO_URL="https://github.com/rishuriya/vad_pi.git"
# TARGET_CLONE_DIR will be defined after user detection

# Paths *relative to the project repository root* after cloning
PYTHON_SCRIPT_REL_PATH="realtime_vad_inference.py" 
REQUIREMENTS_FILE_REL_PATH="requirements_inference.txt" 

# Other paths (will be defined after user detection)
# VENV_PATH
WIFI_HELPER_SCRIPT_PATH="/usr/local/bin/wifi_connect_helper.sh"
SERVICE_DIR="/etc/systemd/system"
WIFI_SERVICE_NAME="wifi-connect"
VAD_SERVICE_NAME="vad-inference"

# --- Script Start ---
echo "Starting Full VAD Inference Setup Script..."
set -e # Exit immediately if a command exits with a non-zero status.

# --- Check Root --- 
if [ "$EUID" -ne 0 ]; then
  echo "🛑 This script needs to be run with sudo privileges."
  exec sudo bash "$0" "$@" 
fi
echo "Running as root..."

# --- Determine Target User --- 
if [ -z "$SUDO_USER" ] || [ "$SUDO_USER" == "root" ]; then
  echo "🚨 Could not determine the original user who ran sudo. Please run like: sudo bash $0" 
  exit 1
fi
TARGET_USER="$SUDO_USER"
echo "Detected target user: $TARGET_USER"

# --- Define User-Specific Paths ---
TARGET_CLONE_DIR="/home/$TARGET_USER/vad_project"
VENV_PATH="/home/$TARGET_USER/venv_vad"
echo "Project directory set to: $TARGET_CLONE_DIR"
echo "Virtual env path set to: $VENV_PATH"

# --- Update Package List ---
echo "Updating package list..."
apt-get update -y

# --- Install Prerequisites ---
echo "Installing prerequisites (git, python3, venv, network-manager, wget, portaudio)..."
apt-get install -y git python3 python3-venv network-manager wget portaudio19-dev || { echo "🚨 Failed to install prerequisites."; exit 1; }

# --- Wi-Fi Connection Logic (from previous script) ---
echo "Checking Wi-Fi connection..."
if nmcli device status | grep -qE '\s+connected\s+'; then
    echo "✅ Already connected to Wi-Fi."
else
    echo "⚠️ Not connected to Wi-Fi. Attempting connection..."
    WIFI_INTERFACE=$(iw dev | awk '$1=="Interface"{print $2}' | head -n1)
    if [ -z "$WIFI_INTERFACE" ]; then
        echo "🚨 Could not detect Wi-Fi interface."
        exit 1
    fi
    echo "Detected Wi-Fi interface: $WIFI_INTERFACE"
    nmcli radio wifi on
    echo "Scanning for networks..."
    sleep 5 
    echo "Trying to connect to known network: $KNOWN_WIFI_SSID..."
    # Use the helper script logic directly here for the initial connection attempt
    if nmcli device wifi connect "$KNOWN_WIFI_SSID" password "$KNOWN_WIFI_PSK" ifname "$WIFI_INTERFACE"; then
        echo "✅ Successfully connected to $KNOWN_WIFI_SSID."
    else
        echo "🚨 Failed to connect to known network: $KNOWN_WIFI_SSID."
        echo "   Will rely on the wifi-connect service after reboot."
        # Don't exit here, let the service handle retries later
    fi
    # Short pause even on failure before proceeding
    sleep 2 
fi

# --- Clone Project Repository ---
echo "Cloning project repository from $GIT_REPO_URL into $TARGET_CLONE_DIR..."
if [ -d "$TARGET_CLONE_DIR" ]; then
    echo "Removing existing project directory: $TARGET_CLONE_DIR"
    rm -rf "$TARGET_CLONE_DIR"
fi
# Clone as the target user
sudo -u "$TARGET_USER" git clone "$GIT_REPO_URL" "$TARGET_CLONE_DIR" || { echo "🚨 Failed to clone repository."; exit 1; }

# --- <<< START Permission Fixes >>> ---
echo "Ensuring correct permissions for $TARGET_USER on $TARGET_CLONE_DIR..."
# Ensure the target user can enter their home directory (should be default, but just in case)
chmod u+x "/home/$TARGET_USER"
# Ensure the target user can read files and enter directories within the cloned repo
chown -R "$TARGET_USER:$TARGET_USER" "$TARGET_CLONE_DIR"
chmod -R u+rX "$TARGET_CLONE_DIR"
echo "✅ Permissions set."
# --- <<< END Permission Fixes >>> ---

# Check if cloning was successful and target directory exists
if [ ! -d "$TARGET_CLONE_DIR" ]; then
    echo "🚨 Target clone directory $TARGET_CLONE_DIR not found after git clone."
    exit 1
fi

# --- Set up Python Virtual Environment --- 
echo "Setting up Python virtual environment at $VENV_PATH..."
if [ -d "$VENV_PATH" ]; then
    echo "✅ Virtual environment already exists at $VENV_PATH. Reusing it."
    # Optional: Add an upgrade pip command here if desired
    # sudo -u "$TARGET_USER" "$VENV_PATH/bin/pip" install --upgrade pip
else
    echo "Creating new virtual environment..."
    # Create venv as the target user
    sudo -u "$TARGET_USER" python3 -m venv "$VENV_PATH" || { echo "🚨 Failed to create virtual environment."; exit 1; }
    echo "✅ New virtual environment created."
fi

REQUIREMENTS_ABS_PATH="$TARGET_CLONE_DIR/$REQUIREMENTS_FILE_REL_PATH"
if [ ! -f "$REQUIREMENTS_ABS_PATH" ]; then echo "🚨 Requirements file not found: $REQUIREMENTS_ABS_PATH"; exit 1; fi

echo "Installing/Updating Python requirements from $REQUIREMENTS_ABS_PATH..."
# Install requirements as the target user (will install or update packages)
sudo -u "$TARGET_USER" "$VENV_PATH/bin/pip" install -r "$REQUIREMENTS_ABS_PATH" || { echo "🚨 Failed to install requirements."; exit 1; }

# --- << START Hugging Face Token Handling >> ---
echo "" # Add some spacing
echo "--- Hugging Face Token Configuration ---"
echo "The VAD inference script requires a Hugging Face access token to download models."
echo "You can create one at https://huggingface.co/settings/tokens (read permissions are sufficient)."

# Loop until a non-empty token is provided
HF_TOKEN_INPUT=""
while [ -z "$HF_TOKEN_INPUT" ]; do
    read -s -p "Enter your Hugging Face Token: " HF_TOKEN_INPUT
    echo # Add a newline after secret input
    if [ -z "$HF_TOKEN_INPUT" ]; then
        echo "⚠️ Token cannot be empty. Please try again."
    fi
done

# --- Configure Environment Variable Persistence ---
# Detect the shell and determine the profile file for the target user
# Note: We are configuring for the target user as the service runs as them
# TARGET_USER is already set above
USER_SHELL=$(getent passwd $TARGET_USER | cut -d: -f7 || echo "/bin/bash") # Default to bash if lookup fails
PROFILE_FILE=""

if [[ "$USER_SHELL" == */bash ]]; then
    PROFILE_FILE="/home/$TARGET_USER/.bashrc"
elif [[ "$USER_SHELL" == */zsh ]]; then
    PROFILE_FILE="/home/$TARGET_USER/.zshrc"
else
    # Fallback for other shells or if detection fails
    PROFILE_FILE="/home/$TARGET_USER/.profile"
    echo "Warning: Detected shell for user '$TARGET_USER' is not Bash or Zsh. Attempting to use $PROFILE_FILE."
fi

echo "Configuring Hugging Face token in $PROFILE_FILE for user '$TARGET_USER'..."

# Check if the file exists, create if not
touch "$PROFILE_FILE"
chown $TARGET_USER:$TARGET_USER "$PROFILE_FILE" # Ensure correct ownership even if created by root

# Use tee with sudo to append (or overwrite if already present) to the user's file
# Remove existing lines first to prevent duplicates/update token
grep -v "export HUGGING_FACE_TOKEN=" "$PROFILE_FILE" > "/tmp/profile_temp" || true # Ignore grep exit code if not found
mv "/tmp/profile_temp" "$PROFILE_FILE"
chown $TARGET_USER:$TARGET_USER "$PROFILE_FILE"
chmod 644 "$PROFILE_FILE" # Standard permissions

# Append the new token
{
    echo "" # Add a newline before the comment
    echo "# Hugging Face Token for VAD Project (added/updated by setup_vad.sh)"
    echo "export HUGGING_FACE_TOKEN=\"$HF_TOKEN_INPUT\""
} >> "$PROFILE_FILE" # Append directly as we own the file now (temporarily)

# Ensure correct ownership again after appending
chown $TARGET_USER:$TARGET_USER "$PROFILE_FILE"

echo "✅ Token export added to $PROFILE_FILE."
echo "   The token will be available in new shells logged in as '$TARGET_USER'."
echo "--- End Hugging Face Token Configuration ---"
echo "" # Add spacing
# --- << END Hugging Face Token Handling >> ---

# --- Create Wi-Fi Connect Helper Script ---
echo "Creating Wi-Fi helper script at $WIFI_HELPER_SCRIPT_PATH..."
cat << EOF > "$WIFI_HELPER_SCRIPT_PATH"
#!/bin/bash
# Configuration
WIFI_SSID="$KNOWN_WIFI_SSID"
WIFI_PSK="$KNOWN_WIFI_PSK"
LOG_TAG="wifi-connect-helper"
# Check current connection status
if nmcli device status | grep -qE '\\s+connected\\s+'; then
    logger -t "\$LOG_TAG" "Already connected to Wi-Fi."
    exit 0
fi
# Attempt to connect if not connected
logger -t "\$LOG_TAG" "Wi-Fi not connected. Attempting to connect to SSID: \$WIFI_SSID..."
WIFI_INTERFACE=\$(iw dev | awk '\$1=="Interface"{print \$2}' | head -n1)
if [ -z "\$WIFI_INTERFACE" ]; then
    logger -t "\$LOG_TAG" "Error: Could not detect Wi-Fi interface."
    exit 1
fi
# Try connecting using nmcli
if nmcli device wifi connect "\$WIFI_SSID" password "\$WIFI_PSK" ifname "\$WIFI_INTERFACE"; then
    logger -t "\$LOG_TAG" "Successfully connected to \$WIFI_SSID."
    exit 0
else
    ERR_CODE=\$?
    logger -t "\$LOG_TAG" "Error: Failed to connect to \$WIFI_SSID (nmcli exit code: \$ERR_CODE)."
    exit 1
fi
EOF

chmod +x "$WIFI_HELPER_SCRIPT_PATH"
echo "✅ Wi-Fi helper script created."

# --- Create wifi-connect Service File ---
echo "Creating systemd service file: ${SERVICE_DIR}/${WIFI_SERVICE_NAME}.service"
cat << EOF > "${SERVICE_DIR}/${WIFI_SERVICE_NAME}.service"
[Unit]
Description=Attempt to connect to known Wi-Fi network ($KNOWN_WIFI_SSID) on boot
Wants=network.target
After=network.target

[Service]
Type=oneshot
ExecStart=$WIFI_HELPER_SCRIPT_PATH
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
echo "✅ wifi-connect service file created."

# --- Create vad-inference Service File ---
# Define absolute paths based on the cloned repo location
PYTHON_EXEC_PATH="$VENV_PATH/bin/python"
PYTHON_SCRIPT_ABS_PATH="$TARGET_CLONE_DIR/$PYTHON_SCRIPT_REL_PATH"
# WORKING_DIR=$(dirname "$PYTHON_SCRIPT_ABS_PATH") # No longer needed, using TARGET_CLONE_DIR directly

# Verify paths needed for the service
if [ ! -f "$PYTHON_EXEC_PATH" ]; then echo "🚨 Python executable not found: $PYTHON_EXEC_PATH"; exit 1; fi
if [ ! -f "$PYTHON_SCRIPT_ABS_PATH" ]; then echo "🚨 Python script not found: $PYTHON_SCRIPT_ABS_PATH"; exit 1; fi
# if [ ! -d "$WORKING_DIR" ]; then echo "🚨 Working directory not found: $WORKING_DIR"; exit 1; fi # No longer needed
# Verify the clone directory exists instead
if [ ! -d "$TARGET_CLONE_DIR" ]; then echo "🚨 Target clone directory not found: $TARGET_CLONE_DIR"; exit 1; fi

# --- <<< START Group Fix >>> ---
# Get the primary group name for the target user
TARGET_GROUP=$(id -gn "$TARGET_USER") || { echo "🚨 Failed to get group name for user $TARGET_USER"; exit 1; }
echo "Detected primary group for $TARGET_USER: $TARGET_GROUP"
# --- <<< END Group Fix >>> ---

# Ensure the Python script is executable
echo "Ensuring realtime_vad_inference.py is executable..."
chmod +x "$TARGET_CLONE_DIR/$PYTHON_SCRIPT_REL_PATH"
chmod 755 "$TARGET_CLONE_DIR/$PYTHON_SCRIPT_REL_PATH"
echo "✅ Python script permissions set."

echo "Creating systemd service file: ${SERVICE_DIR}/${VAD_SERVICE_NAME}.service"
cat << EOF > "${SERVICE_DIR}/${VAD_SERVICE_NAME}.service"
[Unit]
Description=Realtime VAD Inference Service ($VAD_SERVICE_NAME)
# Wait for network connection attempt and sound system
Wants=network-online.target ${WIFI_SERVICE_NAME}.service
After=network-online.target sound.target ${WIFI_SERVICE_NAME}.service

[Service]
User=$TARGET_USER
WorkingDirectory=$TARGET_CLONE_DIR
# Pass the Hugging Face token as an environment variable to the service
Environment="HUGGING_FACE_TOKEN=$HF_TOKEN_INPUT"
ExecStart=$PYTHON_EXEC_PATH $PYTHON_SCRIPT_ABS_PATH
Restart=on-failure
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
echo "✅ vad-inference service file created."

# --- Enable Services ---
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling ${WIFI_SERVICE_NAME}.service to start on boot..."
systemctl enable "${WIFI_SERVICE_NAME}.service"

echo "Enabling ${VAD_SERVICE_NAME}.service to start on boot..."
systemctl enable "${VAD_SERVICE_NAME}.service"

# --- Cleanup ---
# Decide if you want to remove the setup script itself
# Find the script's own path reliably
# SCRIPT_PATH=$(readlink -f "$0") 
# if [ -f "$SCRIPT_PATH" ]; then
#   echo "Removing setup script: $SCRIPT_PATH"
#   rm "$SCRIPT_PATH"
# fi
# Removing the cloned repo might be undesirable if models/etc are needed
# echo "Removing cloned repository directory: $TARGET_CLONE_DIR"
# rm -rf "$TARGET_CLONE_DIR" 

echo "✅ Setup complete! Services enabled."
echo "   Reboot the Raspberry Pi for the services to take effect."
echo "   After reboot, check status with:"
echo "     sudo systemctl status ${WIFI_SERVICE_NAME}.service"
echo "     sudo systemctl status ${VAD_SERVICE_NAME}.service"
echo "   View logs with:"
echo "     sudo journalctl -u ${WIFI_SERVICE_NAME}.service -f"
echo "     sudo journalctl -u ${VAD_SERVICE_NAME}.service -f"

exit 0