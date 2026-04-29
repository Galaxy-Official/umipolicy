#!/usr/bin/env bash

# Change to the monitor directory and run the receiver
cd "$(dirname "$0")/monitor" || exit 1
python monitor_receiver.py
