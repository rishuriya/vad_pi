#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "====================================="
echo "   VAD Pi - One-Click Installer"
echo "====================================="
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}This script needs to be run with sudo privileges.${NC}"
  echo -e "Please run: ${YELLOW}sudo bash -c \"$(curl -fsSL https://raw.githubusercontent.com/rishuriya/vad_pi/main/install_vad_pi.sh)\"${NC}"
  exit 1
fi

echo -e "${YELLOW}This script will:${NC}"
echo "1. Clone the VAD Pi repository"
echo "2. Run the setup process"
echo "3. Configure system services"
echo "4. Ensure proper permissions for all scripts"
echo ""

# Ask for confirmation before proceeding
read -p "Do you want to continue with the installation? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
  echo -e "${YELLOW}Installation cancelled.${NC}"
  exit 0
fi

# Create temporary directory for cloning
TEMP_DIR=$(mktemp -d)
echo -e "\n${GREEN}Creating temporary directory: ${TEMP_DIR}${NC}"

# Clone the repository
echo -e "\n${GREEN}Cloning VAD Pi repository...${NC}"
git clone https://github.com/rishuriya/vad_pi.git "${TEMP_DIR}" || {
  echo -e "${RED}Failed to clone repository. Check your internet connection.${NC}"
  rm -rf "${TEMP_DIR}"
  exit 1
}

# Navigate to cloned repository
cd "${TEMP_DIR}" || {
  echo -e "${RED}Failed to enter repository directory.${NC}"
  rm -rf "${TEMP_DIR}"
  exit 1
}

# Make scripts executable
echo -e "\n${GREEN}Setting executable permissions for scripts...${NC}"
chmod +x setup_vad.sh
chmod +x run_setup.sh
chmod +x realtime_vad_inference.py

# Modify setup_vad.sh to ensure permissions for Python script
echo -e "\n${GREEN}Updating setup script to ensure proper permissions...${NC}"
cat << 'EOF' > permission_fix.patch
# Add these lines to setup_vad.sh before systemd service creation
echo "Ensuring realtime_vad_inference.py is executable..."
chmod +x "$TARGET_CLONE_DIR/$PYTHON_SCRIPT_REL_PATH"
chmod 755 "$TARGET_CLONE_DIR/$PYTHON_SCRIPT_REL_PATH"
echo "✅ Python script permissions set."
EOF

# Insert the permission fix into setup_vad.sh
# Find the line right before service file creation
LINE_NUMBER=$(grep -n "# --- Create vad-inference Service File ---" setup_vad.sh | cut -d: -f1)
if [ -n "$LINE_NUMBER" ]; then
  # Insert before that line
  sed -i "${LINE_NUMBER}r permission_fix.patch" setup_vad.sh
  echo -e "${GREEN}Successfully updated setup script with permission fixes.${NC}"
else
  echo -e "${YELLOW}Could not find insertion point for permission fix. This may cause issues.${NC}"
fi

# Run the setup script
echo -e "\n${GREEN}Running VAD Pi setup...${NC}\n"
./setup_vad.sh

# Check the setup exit status
SETUP_EXIT_CODE=$?
if [ $SETUP_EXIT_CODE -eq 0 ]; then
  # Cleanup the temporary directory
  cd /
  rm -rf "${TEMP_DIR}"
  
  echo -e "\n${GREEN}Setup completed successfully!${NC}"
  echo -e "You should ${YELLOW}reboot your Raspberry Pi${NC} now for all changes to take effect."
  
  # Ask if user wants to reboot now
  read -p "Would you like to reboot now? (y/n): " reboot
  if [[ $reboot == [yY] || $reboot == [yY][eE][sS] ]]; then
    echo "Rebooting system in 5 seconds..."
    sleep 5
    reboot
  else
    echo -e "Remember to reboot manually using ${YELLOW}sudo reboot${NC} when you're ready."
  fi
else
  echo -e "\n${RED}Setup encountered errors. Please check the output above for details.${NC}"
  echo "Temporary files were left at ${TEMP_DIR} for debugging purposes."
  echo "You may need to fix the issues and run the setup again."
fi

exit $SETUP_EXIT_CODE 