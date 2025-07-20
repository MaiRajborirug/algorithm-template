#!/usr/bin/env bash

# Setup script for Docker 28.1.1 on Ubuntu 24.04
echo "Setting up SynthRAD algorithm for Docker 28.1.1 on Ubuntu 24.04..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
TASK_TYPE=mri
INPUT_FOLDER=/input
EOF
    echo ".env file created successfully"
else
    echo ".env file already exists"
fi

# Make scripts executable
chmod +x build.sh
chmod +x test.sh
chmod +x test_gpu.sh
chmod +x export.sh

echo "Setup completed successfully!"
echo ""
echo "To build and test the algorithm:"
echo "1. Run: ./build.sh"
echo "2. Run: ./test.sh"
echo ""
echo "For GPU testing (if available):"
echo "1. Run: ./test_gpu.sh" 