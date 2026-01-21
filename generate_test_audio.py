#!/usr/bin/env python3
"""Generate a simple test audio file for Whisper testing."""

import wave
import struct
import math

# Audio parameters
sample_rate = 16000  # 16 kHz
duration = 2  # 2 seconds
frequency = 440  # A4 note

# Generate audio data (simple sine wave)
num_samples = sample_rate * duration
audio_data = []

for i in range(num_samples):
    # Generate sine wave
    value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
    audio_data.append(struct.pack('<h', value))

# Write WAV file
output_file = 'static/test-audio.wav'
with wave.open(output_file, 'w') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(b''.join(audio_data))

print(f"Created test audio file: {output_file}")
print("This is a 440Hz tone for testing purposes.")
