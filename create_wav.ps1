# Create a minimal valid WAV file for testing
$sampleRate = 16000
$duration = 1
$numSamples = $sampleRate * $duration

# WAV file header
$header = [byte[]](
    0x52, 0x49, 0x46, 0x46,  # "RIFF"
    0x24, 0xF4, 0x01, 0x00,  # File size - 8
    0x57, 0x41, 0x56, 0x45,  # "WAVE"
    0x66, 0x6D, 0x74, 0x20,  # "fmt "
    0x10, 0x00, 0x00, 0x00,  # Subchunk1Size (16)
    0x01, 0x00,              # AudioFormat (PCM)
    0x01, 0x00,              # NumChannels (Mono)
    0x80, 0x3E, 0x00, 0x00,  # SampleRate (16000)
    0x00, 0x7D, 0x00, 0x00,  # ByteRate
    0x02, 0x00,              # BlockAlign
    0x10, 0x00,              # BitsPerSample (16)
    0x64, 0x61, 0x74, 0x61,  # "data"
    0x00, 0xF4, 0x01, 0x00   # Subchunk2Size
)

# Create silent audio data
$audioData = New-Object byte[] $($numSamples * 2)

# Write to file
$outputPath = "static\test-audio.wav"
[System.IO.File]::WriteAllBytes($outputPath, $header + $audioData)

Write-Host "Created test audio file: $outputPath"
