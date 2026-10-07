# AMD Radeon GPU alternative

The Python application uses faster-whisper, whose prebuilt GPU backend supports NVIDIA. `--device cpu` works with AMD Ryzen CPUs; it does not use Radeon graphics.

For GPU acceleration on a Radeon or compatible integrated GPU, use the separate [whisper.cpp Vulkan backend](https://github.com/ggml-org/whisper.cpp#vulkan-gpu-support). This is an external workflow, not a backend implemented by `transcribe.py`. It has not been tested on AMD hardware in this project.

## Windows example

Install a Vulkan-capable graphics driver, [Vulkan SDK](https://vulkan.lunarg.com/sdk/home), CMake, Visual Studio C++ build tools, and FFmpeg. Run the build commands in a Visual Studio Developer PowerShell:

```powershell
git clone https://github.com/ggml-org/whisper.cpp.git vendor/whisper.cpp
cmake -S vendor/whisper.cpp -B vendor/whisper.cpp/build -DGGML_VULKAN=ON
cmake --build vendor/whisper.cpp/build --config Release --parallel
New-Item -ItemType Directory -Force models
Invoke-WebRequest "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" -OutFile "models/ggml-small.bin"
```

From the transcriber folder, prepare a local audio file and run the separately built CLI:

```powershell
New-Item -ItemType Directory -Force downloads,transcripts
ffmpeg -i "C:\Media\lecture.mp4" -ar 16000 -ac 1 -c:a pcm_s16le "downloads/input.wav"
& ".\vendor\whisper.cpp\build\bin\Release\whisper-cli.exe" -m "models/ggml-small.bin" -f "downloads/input.wav" -l es -otxt -osrt -of "transcripts/amd_result"
```

For a YouTube URL, first download one audio stream:

```powershell
.\.venv\Scripts\python.exe -m yt_dlp --js-runtimes node -f bestaudio -o "downloads/input.%(ext)s" "https://www.youtube.com/watch?v=DEpe7VdDfpY"
```

Use the actual downloaded filename with FFmpeg in place of `lecture.mp4`. Check whisper.cpp's startup logs for the selected Vulkan device; a successful CLI run alone does not prove GPU acceleration. Its TXT/SRT output differs from this application's TXT/JSON format.

On Linux, build with the same CMake Vulkan flag, download a GGML model using upstream's shell script, and run `vendor/whisper.cpp/build/bin/whisper-cli`. Supported hardware depends on the driver and Vulkan implementation.

ROCm/HIP is another upstream option with a separate [AMD hardware compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html). Installing ROCm does not add AMD support to the current faster-whisper backend. A Ryzen AI NPU is also a distinct accelerator with separate upstream setup requirements.
