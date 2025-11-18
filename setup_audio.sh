#!/bin/bash
# U-CogNet Audio Dependencies Setup
# Installs audio processing dependencies

set -e

echo "🎵 Setting up U-CogNet Audio Dependencies..."

# Install audio processing libraries
pip install moviepy librosa soundfile

# Optional: Install ffmpeg if not present
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 Installing ffmpeg..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

# Test installations
python -c "import moviepy; print('✅ MoviePy installed')"
python -c "import librosa; print('✅ Librosa installed')"
python -c "import soundfile; print('✅ SoundFile installed')"

echo "✅ Audio dependencies installed successfully"
echo "🎯 Ready for audio cognitive processing"