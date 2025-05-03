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
echo "   VAD Pi - Setup Helper Script"
echo "====================================="
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}This script needs to be run with sudo privileges.${NC}"
  echo -e "Please run: ${YELLOW}sudo bash $0${NC}"
  exit 1
fi

# Check if setup_vad.sh exists
if [ ! -f "setup_vad.sh" ]; then
  echo -e "${RED}Error: setup_vad.sh file not found in the current directory.${NC}"
  echo -e "Make sure you are in the correct directory containing the setup file."
  exit 1
fi

# Make setup_vad.sh executable
chmod +x setup_vad.sh

echo -e "${YELLOW}Important Notes Before Starting:${NC}"
echo "1. This setup will require a Hugging Face access token."
echo "   You can create one at https://huggingface.co/settings/tokens"
echo "2. The setup will configure WiFi and create system services."
echo "3. A reboot will be recommended after setup completes."
echo ""

# Ask for confirmation before proceeding
read -p "Do you want to continue with the setup? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
  echo -e "${YELLOW}Setup cancelled.${NC}"
  exit 0
fi

echo -e "\n${GREEN}Running VAD Pi setup...${NC}\n"

# Run the setup script
./setup_vad.sh

# Check the exit status
if [ $? -eq 0 ]; then
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
  echo "You may need to fix the issues and run the setup again."
fi

exit 0 