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

# Fix the systemd service file to avoid GROUP issues
echo -e "\n${GREEN}Fixing systemd service configuration to address GROUP issues...${NC}"

# Find section in setup_vad.sh that creates the service file
SERVICE_START=$(grep -n "cat << EOF > \"\${SERVICE_DIR}/\${VAD_SERVICE_NAME}.service\"" setup_vad.sh | cut -d: -f1)
SERVICE_END=$(grep -n -A20 "cat << EOF > \"\${SERVICE_DIR}/\${VAD_SERVICE_NAME}.service\"" setup_vad.sh | grep -n "EOF" | head -1 | cut -d: -f1)
SERVICE_END=$((SERVICE_START + SERVICE_END - 1))

if [ -n "$SERVICE_START" ] && [ -n "$SERVICE_END" ]; then
  # Create a backup
  cp setup_vad.sh setup_vad.sh.bak
  
  # Extract the service file definition
  sed -n "${SERVICE_START},${SERVICE_END}p" setup_vad.sh > service_definition.txt
  
  # Create fixed service definition (removing Group=$TARGET_USER line)
  grep -v "Group=\$TARGET_USER" service_definition.txt > service_definition_fixed.txt
  
  # Replace the service definition in the setup_vad.sh
  sed -i "${SERVICE_START},${SERVICE_END}d" setup_vad.sh
  sed -i "${SERVICE_START}r service_definition_fixed.txt" setup_vad.sh
  
  echo -e "${GREEN}Successfully fixed systemd service definition to avoid GROUP errors.${NC}"
else
  echo -e "${YELLOW}Could not find systemd service definition in setup script. Will attempt a workaround.${NC}"
  
  # Create a post-setup hook to fix the service file
  cat << 'EOF' > fix_service.sh
#!/bin/bash
# This script removes the problematic Group line from the vad-inference service file
SERVICE_FILE="/etc/systemd/system/vad-inference.service"
if [ -f "$SERVICE_FILE" ]; then
  echo "Fixing GROUP issue in $SERVICE_FILE..."
  cp "$SERVICE_FILE" "$SERVICE_FILE.bak"
  grep -v "^Group=" "$SERVICE_FILE.bak" > "$SERVICE_FILE"
  systemctl daemon-reload
  echo "✅ Service file fixed."
fi
EOF
  chmod +x fix_service.sh
fi

# Run the setup script
echo -e "\n${GREEN}Running VAD Pi setup...${NC}\n"
./setup_vad.sh

# Check the setup exit status
SETUP_EXIT_CODE=$?
if [ $SETUP_EXIT_CODE -eq 0 ]; then
  # If we created a post-setup hook, run it now
  if [ -f "fix_service.sh" ]; then
    echo -e "\n${GREEN}Applying post-setup fixes for GROUP issue...${NC}"
    ./fix_service.sh
  fi
  
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
  
  # Add a manual fix suggestion
  echo -e "\n${YELLOW}If the issue is related to GROUP errors, you can try this manual fix:${NC}"
  echo "sudo sed -i '/^Group=/d' /etc/systemd/system/vad-inference.service"
  echo "sudo systemctl daemon-reload"
  echo "sudo systemctl restart vad-inference.service"
fi

exit $SETUP_EXIT_CODE 