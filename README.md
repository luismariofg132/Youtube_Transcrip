# YouTube Transcriber

Repository: [luismariofg132/Youtube_Transcrip](https://github.com/luismariofg132/Youtube_Transcrip).

Transcribe YouTube videos and local audio/video files into timestamped UTF-8 text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Speech recognition runs locally, with NVIDIA CUDA or CPU inference. Each transcript includes a JSON processing report.

The code, CLI options, comments, and report fields are in English. Speech is transcribed in its original language; the default is Spanish (`es`). No API key is required.

## Hardware compatibility

| Hardware | This Python application | Setup |
| --- | --- | --- |
| NVIDIA GPU | CUDA inference (`int8_float16`) | [NVIDIA setup](#nvidia-cuda-setup) |
| AMD Ryzen CPU | CPU inference (`int8`) | [CPU setup](#cpu-setup-windows-linux-and-macos) |
| Intel CPU | CPU inference (`int8`) | CPU setup |
| AMD Radeon GPU or Radeon integrated graphics | No GPU backend in faster-whisper; CPU mode works | [AMD GPU alternative](docs/AMD_GPU.md) |
| Intel GPU | No GPU backend in faster-whisper; CPU mode works | CPU mode or the external Vulkan alternative |
| Apple Silicon | CPU mode; this application does not use Metal | CPU setup, subject to dependency wheel availability |

**Ryzen is a processor family.** Some Ryzen processors include Radeon graphics or a Ryzen AI NPU; these are separate devices. An AMD GPU does not run this application's CUDA backend. See [AMD's processor page](https://www.amd.com/en/products/processors/desktops/ryzen.html) and [CTranslate2 hardware support](https://opennmt.net/CTranslate2/hardware_support.html).

`--device auto` selects CUDA when CTranslate2 detects an NVIDIA GPU, otherwise CPU, and prints the selected device. A CUDA library or memory error stops the operation; the application never silently retries it on CPU. Use `--device cpu` explicitly if needed.

## Requirements

- Python **3.12**, 64-bit, is the tested version.
- For YouTube downloads, install a current [Node.js](https://nodejs.org/en/download) runtime and put `node` on PATH. `yt-dlp[default]` installs its JavaScript solver component. See [yt-dlp dependencies](https://github.com/yt-dlp/yt-dlp#dependencies).
- Internet access for package installation, YouTube downloads, and the first download of each model.
- NVIDIA inference additionally requires a compatible driver and CUDA libraries; see below.
- On Windows, install the [Visual C++ runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) if a CTranslate2 import reports a missing Microsoft runtime DLL.

The Git repository contains source code, configuration examples, and documentation. Python, Node.js, dependency binaries, downloaded models, media, and generated transcripts are **not uploaded to Git**. A prepared local Windows folder may already contain these ignored files. A fresh clone must follow the installation steps.

## CPU setup (Windows, Linux, and macOS)

Clone the repository and open its directory:

```bash
git clone https://github.com/luismariofg132/Youtube_Transcrip.git
cd Youtube_Transcrip
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe transcribe.py --diagnose --device cpu
.\.venv\Scripts\python.exe transcribe.py --device cpu "C:\Media\lecture.m4a"
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python transcribe.py --diagnose --device cpu
.venv/bin/python transcribe.py --device cpu "/path/to/lecture.m4a"
```

These steps work with AMD Ryzen or Intel CPUs. CPU transcription is usually slower than NVIDIA inference. Models are automatically downloaded to `models` on first use; cached models can be used offline with local media.

## NVIDIA CUDA setup

Install/update the NVIDIA driver for your exact GPU from [NVIDIA](https://www.nvidia.com/Download/index.aspx), then check `nvidia-smi`. The CUDA number shown there describes driver compatibility, not the presence of every runtime library.

On Windows, create the virtual environment as above, then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe transcribe.py --diagnose --device cuda
.\.venv\Scripts\python.exe transcribe.py --device cuda --model small "C:\Media\lecture.m4a"
```

The tested Windows CTranslate2 4.8.2 wheel includes its cuDNN DLL; `requirements-cuda.txt` installs the NVIDIA CUDA 12 cuBLAS runtime. The application registers installed DLL directories for its own process, without changing the system PATH. If a wheel or installation differs, use the [upstream GPU installation instructions](https://github.com/SYSTRAN/faster-whisper#gpu), which specify CUDA 12 and cuDNN 9.

On Linux, `requirements-cuda.txt` also installs cuDNN 9. Set the library search path **before** launching Python:

```bash
.venv/bin/python -m pip install -r requirements-cuda.txt
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import nvidia.cublas, nvidia.cudnn; print(str(nvidia.cublas.__path__[0]) + "/lib:" + str(nvidia.cudnn.__path__[0]) + "/lib")')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python transcribe.py --diagnose --device cuda
```

Linux installation is documented from upstream instructions; the GPU integration tests for this repository were performed on Windows. Start with `small` and `--batch-size 1` on a 4 GB GPU. Larger models and batches require more VRAM.

## Usage

Examples below assume `python` refers to the installed environment. On Windows you can use `.\.venv\Scripts\python.exe` explicitly instead.

```bash
python transcribe.py "https://www.youtube.com/watch?v=DEpe7VdDfpY"
python transcribe.py "/path/to/lecture.mp4" --device cpu
python transcribe.py --list examples/urls.txt --language es
python transcribe.py --language auto "/path/to/audio.wav"
python transcribe.py --device cuda --batch-size 4 "/path/to/audio.m4a"
python transcribe.py --no-timestamps --output-dir transcripts "/path/to/audio.mp3"
```

Replace the example URL with any YouTube video you want to transcribe. To process several sources, pass multiple quoted arguments or put one source per line in a UTF-8 list file. Blank lines and `#` comments are ignored. `examples/urls.txt` contains the same example video; replace or extend it as needed. Relative file paths are resolved from the console's current working directory. Videos are processed in full; playlists and channels are not supported.

On Windows, double-click `TRANSCRIBE.cmd`, enter one or more sources, and submit an empty line to start. You can also drag local media onto it. Helpers:

| Launcher | Purpose |
| --- | --- |
| `INSTALL.cmd` | Create `.venv` and install CPU dependencies |
| `INSTALL.cmd cuda` | Install NVIDIA dependencies |
| `TRANSCRIBE.cmd` | Interactive or argument-based transcription |
| `DIAGNOSE.cmd --device cpu` | Real CPU inference check |
| `DIAGNOSE.cmd --device cuda` | Real CUDA inference check |
| `UPDATE_YOUTUBE.cmd` | Update the downloader when YouTube changes |

The launcher prefers `.venv`, then an optional ignored `python_local.txt`, a local `python` bundle, and finally a system Python. An active virtual environment takes precedence over local bundled dependencies.

## Configuration and outputs

Copy `config.example.json` to `config.json` to customize defaults. The real configuration is ignored by Git. CLI options override it.

```json
{
  "model": "small",
  "language": "es",
  "device": "auto",
  "batch_size": 1,
  "cpu_threads": 8
}
```

Downloads go to `downloads`. Results go to `transcripts` unless `--output-dir` is specified. Every run creates a unique TXT and JSON. A `.txt.partial` file preserves recovered text if processing fails or is cancelled; it becomes a final TXT only after completion. Existing transcripts are preserved.

Reports contain the actual device, compute type, language, duration, segment count, and timing. `transcription_seconds` includes decoding and speech detection but excludes model loading and YouTube download. `total_seconds` includes the download but still excludes model loading.

Recognition and timestamps are approximate. Names, technical terms, overlapping speech, and quiet recordings may be misrecognized. Speaker identification is not implemented. Some YouTube videos require authentication or block downloads; provide local media in those cases. The application does not extract browser cookies or log into accounts.

## Development and validation

```bash
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
python -m ruff check transcribe.py runtime.py tests
python -m ruff format --check transcribe.py runtime.py tests
```

Offline regression tests cover URL validation, Windows filenames, partial transcript preservation, UTF-8 output, repeated runs, and CUDA failures without silent CPU fallback. GitHub Actions runs these tests and lint checks on Windows and Linux. Hardware inference is checked separately using `--diagnose` and local media; CI does not claim GPU validation.

The original Windows benchmark used an RTX 3050 Laptop GPU with 4 GB VRAM, model `small`, `int8_float16`, batch size 1. A 39:47 recording took 94.4 seconds from a YouTube download and 99.5 seconds from the supplied M4A (processing only). These are observed results, not performance guarantees. AMD GPU inference has not been tested on this machine.

## Publish your repository

To publish changes to [Youtube_Transcrip](https://github.com/luismariofg132/Youtube_Transcrip), commit them and push from the repository directory:

```bash
git add .
git commit -m "Describe your change"
git push origin main
```

Cloning configures `origin` automatically. For an existing local repository without a remote, run `git remote add origin https://github.com/luismariofg132/Youtube_Transcrip.git` once, then use `git push -u origin main` for the first push. `.gitignore` keeps local runtimes, models, media, transcripts, and private settings out of the commit.

## License

This application's source code is licensed under MIT. Third-party tools, model weights, and dependency binaries retain their own licenses.
