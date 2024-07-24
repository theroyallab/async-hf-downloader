"""
One-file asynchronous HuggingFace downloader.

Created by: kingbri
This piece of software should be credited whenever it's used anywhere else.

Requirements: huggingface_hub, aiohttp, aiofiles, rich

Version: 0.0.5
"""

import asyncio
import argparse
import shutil
import aiohttp
import aiofiles
import math
import pathlib
from fnmatch import fnmatch
from huggingface_hub import HfApi, hf_hub_url
from rich.progress import Progress
from typing import List, Optional


def unwrap(wrapped, default=None):
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default

    return wrapped


async def download_file(
    session: aiohttp.ClientSession,
    repo_item: dict,
    token: Optional[str],
    download_path: pathlib.Path,
    chunk_limit: int,
    progress: Progress,
):
    filename = repo_item.get("filename")
    url = repo_item.get("url")

    # Default is 2MB
    chunk_limit_bytes = math.ceil(unwrap(chunk_limit, 2000000) * 100000)

    filepath = download_path / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    req_headers = {"Authorization": f"Bearer {token}"} if token else {}

    async with session.get(url, headers=req_headers) as response:
        # TODO: Change to raise errors
        assert response.status == 200

        file_size = int(response.headers["Content-Length"])

        download_task = progress.add_task(
            f"[cyan]Downloading {filename}", total=file_size
        )

        # Chunk limit is 2 MB
        async with aiofiles.open(str(filepath), "wb") as f:
            async for chunk in response.content.iter_chunked(chunk_limit_bytes):
                await f.write(chunk)
                progress.update(download_task, advance=len(chunk))


# Huggingface does not know how async works
def get_repo_info(repo_id, revision, token):
    api_client = HfApi()
    repo_tree = api_client.list_repo_files(repo_id, revision=revision, token=token)
    return list(
        map(
            lambda filename: {
                "filename": filename,
                "url": hf_hub_url(repo_id, filename, revision=revision),
            },
            repo_tree,
        )
    )


def check_exclusions(
    filename: str, include_patterns: List[str], exclude_patterns: List[str]
):
    include_result = any(fnmatch(filename, pattern) for pattern in include_patterns)
    exclude_result = any(fnmatch(filename, pattern) for pattern in exclude_patterns)

    return include_result and not exclude_result


async def entrypoint(args):
    # Fetches the file list in a separate thread (since hf_hub isn't async)
    file_list = await asyncio.to_thread(
        get_repo_info, args.repo_id, args.revision, args.token
    )

    if args.include or args.exclude:
        include_patterns = unwrap(args.include, ["*"])
        exclude_patterns = unwrap(args.exclude, [])

        file_list = [
            file
            for file in file_list
            if check_exclusions(
                file.get("filename"), include_patterns, exclude_patterns
            )
        ]

    if not args.skip_checks and not file_list:
        raise ValueError(
            f"File list for repo {args.repo_id} is empty. Check your filters?"
        )

    download_path = pathlib.Path(
        unwrap(args.download_path, args.repo_id.split("/")[-1])
    )
    download_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_checks and download_path.exists():
        raise FileExistsError(
            f"The path {download_path} already exists. Remove the folder and try again."
        )

    print(f"Saving to {str(download_path)}")

    try:
        client_timeout = aiohttp.ClientTimeout(total=args.timeout) # Turn off timeout
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            tasks = []
            print(f"Starting download for {args.repo_id}")

            progress = Progress()
            progress.start()

            for repo_item in file_list:
                tasks.append(
                    download_file(
                        session,
                        repo_item,
                        token=args.token,
                        download_path=download_path.resolve(),
                        chunk_limit=args.chunk_limit,
                        progress=progress,
                    )
                )

            await asyncio.gather(*tasks)
            progress.stop()
            print(f"Finished download for {args.repo_id}")
    except (Exception, asyncio.CancelledError) as exc:
        # Cleanup on exception
        if download_path.is_dir():
            shutil.rmtree(download_path)
        else:
            download_path.unlink()

        # Stop the progress bar
        progress.stop()

        # Re-raise the exception if the task wasn't cancelled
        if not isinstance(exc, asyncio.CancelledError):
            raise exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronous HuggingFace downloader")
    parser.add_argument("repo_id", type=str, help="Repo ID from huggingface")
    parser.add_argument("-r", "--revision", type=str, help="Branch in the repo to use")
    parser.add_argument(
        "-p", "--download-path", type=str, help="Folder name for the model"
    )
    parser.add_argument(
        "-t", "--token", type=str, help="HuggingFace token for private repos"
    )
    parser.add_argument(
        "-i", "--include", nargs="*", type=str, help="Glob patterns of files to include"
    )
    parser.add_argument(
        "-e", "--exclude", nargs="*", type=str, help="Glob patterns of files to exclude"
    )
    parser.add_argument(
        "-c",
        "--chunk-limit",
        type=float,
        help="Override the max limit for download chunks in MB",
    )
    parser.add_argument(
        "-sc",
        "--skip-checks",
        action="store_true",
        help=(
            "Skips all sanity checks such as "
            "checking if the destination directory exists"
        ),
    ),
    parser.add_argument("--timeout", type=int, help="Optional request timeout in seconds")

    args = parser.parse_args()

    asyncio.run(entrypoint(args))
