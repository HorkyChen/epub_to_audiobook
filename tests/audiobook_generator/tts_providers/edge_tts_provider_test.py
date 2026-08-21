import asyncio
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from pydub import AudioSegment

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider
from audiobook_generator.tts_providers.edge_tts_provider import (
    CommWithPauses,
    EdgeTTSProvider,
    TICKS_PER_MS,
    _pcm_bytes_to_ticks,
)
from tests.test_utils import get_edge_config

# 24kHz/16-bit/mono silence, same format CommWithPauses works with internally.
SAMPLE_WIDTH = 2
FRAME_RATE = 24000


def make_silent_mp3_bytes(duration_ms: int) -> bytes:
    """Build a tiny real mp3 so AudioSegment.from_mp3() can decode it in tests."""
    silence = AudioSegment.silent(duration_ms, FRAME_RATE)
    buf = io.BytesIO()
    silence.export(buf, format="mp3")
    return buf.getvalue()


class TestEdgeTtsProvider(unittest.TestCase):

    def test_get_tts_provider(self):
        config = get_edge_config()
        tts_provider = get_tts_provider(config)
        self.assertIsInstance(tts_provider, EdgeTTSProvider)

    def test_default_args(self):
        config = get_edge_config()
        config.voice_name = None
        config.output_format = None
        tts_provider = get_tts_provider(config)
        self.assertEqual(tts_provider.config.voice_name, "en-US-GuyNeural")
        self.assertEqual(tts_provider.config.output_format, "audio-24khz-48kbitrate-mono-mp3")

    def test_pcm_bytes_to_ticks(self):
        # 1 second of 24kHz/16-bit/mono PCM == 48000 bytes == 10_000_000 ticks (100ns units).
        one_second_bytes = SAMPLE_WIDTH * FRAME_RATE
        self.assertEqual(_pcm_bytes_to_ticks(one_second_bytes), 10_000_000)
        self.assertEqual(_pcm_bytes_to_ticks(0), 0)

    def test_chunkify_offsets_events_across_parts_and_pauses(self):
        """Two break_string-separated parts, each reporting a SentenceBoundary at
        offset 0 in their own stream, must end up offset by the real duration of
        the previous part's audio plus the inserted pause, once merged."""
        part_duration_ms = 500
        break_duration_ms = 1250
        part_audio_bytes = b"\x00\x00" * int(FRAME_RATE * part_duration_ms / 1000)  # raw PCM

        async def fake_generate_audio(_self, text):
            return part_audio_bytes, [{"offset": 0, "duration": _pcm_bytes_to_ticks(len(part_audio_bytes)), "text": text}]

        comm = CommWithPauses(
            text="part one @BRK# part two",
            voice_name="en-US-GuyNeural",
            break_string="@BRK#",
            break_duration=break_duration_ms,
            collect_subtitles=True,
        )

        with patch.object(CommWithPauses, "generate_audio", fake_generate_audio):
            asyncio.run(comm.chunkify())

        self.assertEqual(len(comm.cue_events), 2)
        self.assertEqual(comm.cue_events[0]["offset"], 0)
        expected_second_offset = _pcm_bytes_to_ticks(len(part_audio_bytes)) + break_duration_ms * TICKS_PER_MS
        self.assertEqual(comm.cue_events[1]["offset"], expected_second_offset)

    def test_text_to_speech_writes_srt_when_enabled(self):
        config = get_edge_config()
        config.output_srt = True
        provider = get_tts_provider(config)

        events_by_call = [
            [{"offset": 0, "duration": 20_000_000, "text": "Hello world."}],
            [{"offset": 0, "duration": 10_000_000, "text": "Second chunk."}],
        ]

        class FakeCommWithPauses:
            call_count = 0

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.index = FakeCommWithPauses.call_count
                FakeCommWithPauses.call_count += 1
                self.cue_events = events_by_call[self.index]
                self.elapsed_ticks = events_by_call[self.index][0]["duration"]

            async def get_audio_stream(self):
                return io.BytesIO(make_silent_mp3_bytes(50))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "0001_chapter.mp3")
            audio_tags = AudioTags("Chapter 1", "Author", "Book", 1)

            with patch(
                "audiobook_generator.tts_providers.edge_tts_provider.CommWithPauses",
                FakeCommWithPauses,
            ), patch(
                "audiobook_generator.tts_providers.edge_tts_provider.split_text",
                return_value=["chunk one", "chunk two"],
            ):
                provider.text_to_speech("irrelevant, split_text is mocked", output_file, audio_tags)

            srt_path = os.path.join(tmp_dir, "0001_chapter.srt")
            self.assertTrue(os.path.exists(srt_path))
            with open(srt_path, encoding="utf-8") as f:
                srt_content = f.read()
            self.assertIn("Hello world.", srt_content)
            self.assertIn("Second chunk.", srt_content)
            # second cue must start at/after the first chunk's total duration (20_000_000 ticks == 2s)
            self.assertIn("00:00:02", srt_content)

    def test_text_to_speech_skips_srt_when_disabled(self):
        config = get_edge_config()
        config.output_srt = False
        provider = get_tts_provider(config)

        captured_kwargs = {}

        class FakeCommWithPauses:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                self.cue_events = []
                self.elapsed_ticks = 0

            async def get_audio_stream(self):
                return io.BytesIO(make_silent_mp3_bytes(50))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "0001_chapter.mp3")
            audio_tags = AudioTags("Chapter 1", "Author", "Book", 1)

            with patch(
                "audiobook_generator.tts_providers.edge_tts_provider.CommWithPauses",
                FakeCommWithPauses,
            ), patch(
                "audiobook_generator.tts_providers.edge_tts_provider.split_text",
                return_value=["chunk one"],
            ):
                provider.text_to_speech("irrelevant, split_text is mocked", output_file, audio_tags)

            self.assertFalse(os.path.exists(os.path.join(tmp_dir, "0001_chapter.srt")))
            # collect_subtitles=False must have been passed down to CommWithPauses
            self.assertFalse(captured_kwargs.get("collect_subtitles"))


if __name__ == '__main__':
    unittest.main()
