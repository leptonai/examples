"""This example demonstrates how to run optimized Whisper model on
Lepton.

[whisper-jax](https://github.com/sanchit-gandhi/whisper-jax.git) is a
JAX (optimized) port of the openai whisper model. It chunks audio data
into segments and then performs batch inference to gain speedup.

Installing JAX is a bit tricky, so here we provide a combination of
jax + jaxlib + cuda/cudnn pip versions that can work together inside
Lepton's default image.

Whisper has a set of model ids that you can use. This is specified by an
environment variable "WHISPER_MODEL_ID". By default, it uses "openai/whisper-large-v2".
The list of available models are "openai/whisper-{size}" where size can be one of
the following:
    tiny, base, small, medium, large, large-v2
See https://github.com/sanchit-gandhi/whisper-jax for more details.

Optionally, you can also set the environment variable "BATCH_SIZE" to
change the batch size of the inference. By default, it is 4.

In addition, this example also demonstrates how to use Slack bot to
trigger inference. To use this feature, you need to set the following
environment variables:
- `SLACK_VERIFICATION_TOKEN`: The verification token of your Slack app
- `SLACK_BOT_TOKEN`: The bot token of your Slack app
"""

from datetime import datetime, timedelta
import os
import tempfile
from typing import Optional, Dict, Any

from loguru import logger
import requests

from leptonai.photon import Photon, HTTPException


class Whisper(Photon):
    """
    A photon implementatio
    """

    # note:
    requirement_dependency = [
        "git+https://github.com/sanchit-gandhi/whisper-jax.git@0d3bc54",
        "cached_property",
        "nvidia-cudnn-cu11==8.6.0.163",
        "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        "jax==0.4.13",
        "jaxlib==0.4.13+cuda11.cudnn86",
        "slack_sdk",
    ]

    # note: system_dependency specifies what should be installed via `apt install`
    system_dependency = [
        "ffmpeg",
    ]

    def init(self):
        # Implementation note: strictly speaking, this is not recommended by Python
        # as all imports should be places on the top of the file. However, this shows
        # a small trick when a local installation isn't really possible, such as
        # installing all the jax and cuda dependencies on a mac machine. We can defer
        # the import inside the actual Photon class.
        # Of course, this makes the debugging duty to the remote execution time, and
        # is going to be a bit harder. This is a conscious tradeoff between development
        # speed and debugging speed.
        logger.info("Initializing Whisper model. This might take a while...")
        from whisper_jax import FlaxWhisperPipline
        import jax.numpy as jnp

        model_id = os.environ.get("WHISPER_MODEL_ID", "openai/whisper-large-v2")
        batch_size = os.environ.get("BATCH_SIZE", 4)
        logger.info(f"Using model id: {model_id} and batch size: {batch_size}")
        self.pipeline = FlaxWhisperPipline(
            model_id, dtype=jnp.float16, batch_size=batch_size
        )
        logger.info("Initialized Whisper model.")
        logger.info("Initializing slack bot...")
        self._init_slack_bot()

    def _init_slack_bot(self):
        """
        Initializes the slack bot client.
        """
        from slack_sdk import WebClient as SlackClient

        self._verification_token = os.environ.get("SLACK_VERIFICATION_TOKEN", None)
        self._slack_bot_token = os.environ.get("SLACK_BOT_TOKEN", None)
        if self._slack_bot_token:
            self._slack_bot_client = SlackClient(token=self._slack_bot_token)
        else:
            logger.warning("Slack bot token not configured. Slack bot will not work.")
        self._processed_slack_tasks = {}

    @Photon.handler(
        "run",
        example={
            "inputs": (
                "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
            )
        },
    )
    def run(self, inputs: str, task: Optional[str] = None) -> str:
        """
        Transcribe or translate an audio input file to a text transcription.

        Args:
            inputs: the filename or url of the audio file.
            task (optional): either `"transcribe"` or `"translate"`. Defaults to `"transcribe"`.

        Returns:
            text: the transcription of the audio file.
        """
        return self.pipeline(inputs, task=task)["text"]

    async def _slack_process_task(
        self, channel: str, thread_ts: Optional[str], url: str
    ):
        """
        Internal method to process a slack task. This is called by the `slack` handler.
        """
        last_processed_time = self._processed_slack_tasks.get((channel, url))
        if last_processed_time and datetime.now() - last_processed_time < timedelta(
            seconds=20
        ):
            logger.info(
                f"Skip processing slack task: ({channel}, {url}) since it was processed"
                f" recently: {last_processed_time}"
            )
            return

        logger.info(f"Processing audio file: {url}")
        with tempfile.NamedTemporaryFile("wb", suffix="." + url.split(".")[-1]) as f:
            logger.info(f"Start downloading audio file to: {f.name}")
            res = requests.get(
                url,
                allow_redirects=True,
                headers={"Authorization": f"Bearer {self._slack_bot_token}"},
            )
            res.raise_for_status()
            logger.info(f"Downloaded audio file (total bytes: {len(res.content)})")
            f.write(res.content)
            f.flush()
            f.seek(0)
            logger.info(f"Saved audio file to: {f.name}")
            logger.info(f"Running inference on audio file: {f.name}")
            try:
                text = self.run(f.name)
            except Exception:
                logger.error(f"Failed to run inference on audio file: {f.name}")
                return
            logger.info(f"Finished inference on audio file: {f.name}")
        self._slack_bot_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=text,
        )
        if len(self._processed_slack_tasks) > 100:
            # clean up task histories that are too old.
            self._processed_slack_tasks = {
                k: v
                for k, v in self._processed_slack_tasks.items()
                if datetime.now() - v < timedelta(seconds=20)
            }
        self._processed_slack_tasks[(channel, url)] = datetime.now()

    # This is a handler that receives slack events. It is triggered by the
    # slack server side - see the slack event api for details:
    # https://api.slack.com/apis/connections/events-api#receiving-events
    # and more specs here:
    # https://github.com/slackapi/slack-api-specs
    @Photon.handler
    def slack(
        self,
        token: str,
        type: str,
        event: Optional[Dict[str, Any]] = None,
        challenge: Optional[str] = None,
        **exttra,
    ) -> str:
        # Checks if the slack bot is configured. Note that we are still using the
        # now deprecated verification token, an supporting the new signing secrets
        # is a todo item.
        if not self._verification_token or not self._slack_bot_token:
            raise HTTPException(401, "Slack bot not configured.")
        # If token is not correct, we return 401.
        if token != self._verification_token:
            raise HTTPException(401, "Invalid token.")
        # We will respond to the challenge request if it is a url_verification event,
        # so that slack can verify our endpoint.
        if type == "url_verification":
            if challenge:
                return challenge
            else:
                raise HTTPException(400, "Missing challenge")

        # If event is not present, we will ignore it.
        if not event:
            raise HTTPException(400, "Missing event type")

        # Actually handle the slack event. We will only handle file_shared events.
        event_type = event["type"]
        logger.info(f"Received slack event: {event_type}")
        if event_type == "file_shared":
            channel = event["channel_id"]
            thread_ts = event.get("thread_ts")
            file_id = event["file_id"]
            file_info = self._slack_bot_client.files_info(file=file_id)
            if not file_info["ok"]:
                raise HTTPException(500, "Failed to get file info from slack")
            self.add_background_task(
                self._slack_process_task,
                channel,
                thread_ts,
                file_info["file"]["url_private"],
            )
            return "ok"
        else:
            logger.info(f"Ignored slack event type: {event_type}")
            return "ok"


if __name__ == "__main__":
    w = Whisper()
    w.launch()
