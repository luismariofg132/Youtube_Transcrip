# Contributing

Use Python 3.12 in a virtual environment. Keep code, CLI options, comments, and report fields in English.

Run the offline tests and Ruff checks documented in README.md before proposing changes. Do not commit downloaded media, transcript contents, model weights, dependency binaries, credentials, or machine-specific paths.

For GPU issues, include your operating system, Python version, GPU model, driver version, dependency versions, and the output of `--diagnose`. Remove private media paths from logs. Clearly distinguish CPU, CUDA, and external whisper.cpp results; AMD support must be tested on actual AMD hardware before claiming it works.
