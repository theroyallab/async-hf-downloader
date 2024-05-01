# Async HF Downloader

A standalone version of [TabbyAPI's downloader](https://github.com/theroyallab/tabbyAPI/blob/main/common/downloader.py)

## Why did I make this?

I wanted to add a downloader in TabbyAPI, which is a backend for inferencing with LLMs. However, there were some problems with already existing implementations that use the huggingface_hub package and its snapshot_download function.

The main problem was that a KeyboardInterrupt would not stop the download (and therefore won't allow the API to close on SIGINT). In addition, TabbyAPI is an asynchronous program (like most modern networking programs in python). Therefore, huggingface_hub wouldn't fit the bill for a downloader that easily slotted into my project.

Therefore, this project was born and aims to bring an easier experience for users to download their models from huggingface.

## Running

The project can be run in multiple ways:

1. Download the exe or binary from [Releases](https://github.com/theroyallab/async-hf-downloader/releases)
    1. On Windows: `.\async-hf-downloader-win.exe --help`

    2. On Linux: `./async-hf-downloader-linux --help`

    3. On macOS: `./async-hf-downloader-darwin --help`

2. Download the [python file](https://github.com/theroyallab/async-hf-downloader/blob/main/async-hf-downloader/download.py) and run it in your existing project

3. Run as a python package:
    1. Clone the repository

    2. Run `pip install .` in a venv

    3. Run `python -m async-hf-downloader.download --help`

## Contributing

If you have issues with the project:

- Describe the issue in detail

- If you have a feature request, please indicate it as such.

If you have a Pull Request

- Describe the pull request in detail, what, and why you are changing something

## Developers and Permissions

Creators/Developers:

- [kingbri](https://github.com/bdashore3)
