#!/bin/bash

# Configuration (Replace with your actual details)
WIFI_SSID="YOUR_WIFI_SSID"
WIFI_PSK="YOUR_WIFI_PASSWORD"
LOG_TAG="wifi-connect-helper" # For system logs

# Check current connection status
if nmcli device status | grep -qE '\s+connected\s+'; then
    logger -t "$LOG_TAG" "Already connected to Wi-Fi."
    exit 0 # Success, already connected
fi

# Attempt to connect if not connected
logger -t "$LOG_TAG" "Wi-Fi not connected. Attempting to connect to SSID: $WIFI_SSID..."
WIFI_INTERFACE=$(iw dev | awk '$1=="Interface"{print $2}' | head -n1)

if [ -z "$WIFI_INTERFACE" ]; then
    logger -t "$LOG_TAG" "Error: Could not detect Wi-Fi interface."
    exit 1
fi

# Try connecting using nmcli
if nmcli device wifi connect "$WIFI_SSID" password "$WIFI_PSK" ifname "$WIFI_INTERFACE"; then
    logger -t "$LOG_TAG" "Successfully connected to $WIFI_SSID."
    exit 0 # Success
else
    ERR_CODE=$?
    logger -t "$LOG_TAG" "Error: Failed to connect to $WIFI_SSID (nmcli exit code: $ERR_CODE)."
    exit 1 # Failure
fi