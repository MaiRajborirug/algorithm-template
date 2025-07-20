# Docker 28.1.1 Compatibility Guide

This guide provides instructions for running the SynthRAD algorithm template with Docker 28.1.1 on Ubuntu 24.04.

## Prerequisites

- Docker 28.1.1 or later
- Ubuntu 24.04 or later
- At least 4GB of available RAM (configurable in test scripts)

## Quick Start

1. **Setup the environment:**
   ```bash
   ./setup_docker_.sh
   ```

2. **Build the Docker image:**
   ```bash
   ./build.sh
   ```

3. **Run the tests:**
   ```bash
   ./test.sh
   ```

## Docker Version Compatibility

The repository has been tested and adjusted for compatibility with:
- Docker 28.1.1
- Ubuntu 24.04
- Python 3.12-slim base image

## Key Changes Made

1. **Removed .env file dependency:** The Dockerfile no longer requires a .env file to be present during build
2. **Environment variable fallbacks:** The base algorithm now uses sensible defaults when .env is not present
3. **Updated base image:** Uses python:3.12-slim which is compatible with Ubuntu 24.04
4. **Docker command compatibility:** All docker run commands are compatible with Docker 28.1.1

## Memory Configuration

The default memory limit is set to 4GB. You can modify this in the test scripts:

- `test.sh`: Line 8 - `MEM_LIMIT="4g"`
- `test_gpu.sh`: Line 8 - `MEM_LIMIT="4g"`

Maximum supported memory limit is 30GB (configurable in Grand Challenge settings).

## GPU Support

For GPU testing, ensure you have:
- NVIDIA Docker runtime installed
- Compatible NVIDIA drivers
- Run: `./test_gpu.sh`

## Troubleshooting

### Common Issues

1. **Permission denied errors:**
   ```bash
   chmod +x *.sh
   ```

2. **Docker daemon not running:**
   ```bash
   sudo systemctl start docker
   ```

3. **Insufficient memory:**
   - Reduce `MEM_LIMIT` in test scripts
   - Ensure sufficient system RAM

4. **Volume mount issues:**
   - Ensure Docker has permission to create volumes
   - Check if SELinux is blocking mounts (if applicable)

### Docker Commands Used

The repository uses standard Docker commands that are compatible with Docker 28.1.1:

- `docker build`: Builds the algorithm image
- `docker run`: Runs the algorithm with security constraints
- `docker volume create`: Creates temporary volumes for testing
- `docker volume rm`: Cleans up volumes (commented out in scripts)

## Security Features

The Docker run commands include several security features:

- `--network="none"`: No network access
- `--cap-drop="ALL"`: Drops all capabilities
- `--security-opt="no-new-privileges"`: Prevents privilege escalation
- `--memory` and `--memory-swap`: Memory limits
- `--shm-size`: Shared memory limits
- `--pids-limit`: Process limits

## Testing

The test scripts perform the following:

1. Build the Docker image
2. Create a temporary volume
3. Run the algorithm with test data
4. Validate the output against expected results
5. Clean up resources

## Support

For issues specific to Docker 28.1.1 or Ubuntu 24.04, please check:
- Docker documentation: https://docs.docker.com/
- Ubuntu 24.04 release notes
- Grand Challenge documentation for algorithm requirements 