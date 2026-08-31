"""
This Module is responsible for handling the audio data layer.
"""
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.utils.logger import logger

DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60
DEFAULT_MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # 100 MiB


class AudioDataLayer:
    def __init__(self, config):
        """
        Initialize the Audio Data Layer.
        :param config: The configuration object containing model and device info.
        """
        self.debug = config.get('debug')
        audio_config = config.get('audio') or {}
        self.download_timeout_seconds = audio_config.get(
            'download_timeout_seconds',
            DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        )
        self.max_download_bytes = audio_config.get(
            'max_download_bytes',
            DEFAULT_MAX_DOWNLOAD_BYTES,
        )

    def resolve_audio_source(self, url: str) -> dict:
        """
        Resolve a URL or local path to a local file path for ffmpeg.

        Returns:
            {'path': str, 'is_temporary': bool} on success, or {'error': str}.
            Temporary downloads must be deleted by the caller.
        """
        try:
            parsed_url = urlparse(url)

            if bool(parsed_url.scheme) and bool(parsed_url.netloc):
                if self.debug:
                    logger.debug(
                        f"[debug] [Data Layer] [AudioDataLayer] [resolve_audio_source] "
                        f"Downloading audio file from URL: {url}"
                    )
                return self._download_to_tempfile(url, parsed_url)

            if os.path.exists(url) and os.path.isfile(url):
                if self.debug:
                    logger.debug(
                        f"[debug] [Data Layer] [AudioDataLayer] [resolve_audio_source] "
                        f"Using local audio file: {url}"
                    )
                return {'path': url, 'is_temporary': False}

            error_message = 'Provided url is neither a valid URL nor a valid file path.'
            logger.error(f"[error] [Data Layer] [AudioDataLayer] [resolve_audio_source] {error_message}")
            return {'error': error_message}

        except Exception as e:
            logger.error(
                f"[error] [Data Layer] [AudioDataLayer] [resolve_audio_source] "
                f"An unexpected error occurred: {str(e)}"
            )
            return {'error': 'An unexpected error occurred while processing the request.'}

    def _download_to_tempfile(self, url: str, parsed_url) -> dict:
        """
        Stream-download a remote media file with timeout and size limit.
        """
        suffix = Path(parsed_url.path).suffix or '.bin'
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        try:
            with requests.get(
                url,
                stream=True,
                timeout=self.download_timeout_seconds,
            ) as response:
                if response.status_code != 200:
                    error_message = (
                        f'An error occurred during the HTTP request: '
                        f'HTTP status: {response.status_code}'
                    )
                    logger.error(
                        f"[error] [Data Layer] [AudioDataLayer] [_download_to_tempfile] {error_message}"
                    )
                    self._safe_remove(temp_path)
                    return {'error': error_message}

                downloaded_bytes = 0
                with open(temp_path, 'wb') as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > self.max_download_bytes:
                            error_message = (
                                f'Downloaded file exceeds max size of '
                                f'{self.max_download_bytes} bytes.'
                            )
                            logger.error(
                                f"[error] [Data Layer] [AudioDataLayer] [_download_to_tempfile] {error_message}"
                            )
                            self._safe_remove(temp_path)
                            return {'error': error_message}
                        temp_file.write(chunk)

            return {'path': temp_path, 'is_temporary': True}

        except requests.exceptions.RequestException as req_err:
            self._safe_remove(temp_path)
            logger.error(
                f"[error] [Data Layer] [AudioDataLayer] [_download_to_tempfile] "
                f"HTTP request error: {str(req_err)}"
            )
            return {'error': f'An error occurred during the HTTP request: {str(req_err)}'}
        except Exception:
            self._safe_remove(temp_path)
            raise

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
