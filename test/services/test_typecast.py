import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from app.services import voice


def _ok_response(words):
    response = Mock(status_code=200, content=b"{}")
    response.json.return_value = {
        "audio": base64.b64encode(b"fake-mp3").decode(),
        "audio_format": "mp3",
        "audio_duration": 2.0,
        "words": words,
    }
    return response


class _FakeClip:
    duration = 2.0

    def __init__(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


class TestTypecastTTS(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(voice.os.environ, {"TYPECAST_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_voice_name_routing(self):
        self.assertTrue(voice.is_typecast_voice("typecast:tc_abc"))
        self.assertFalse(voice.is_typecast_voice("ko-KR-SunHiNeural"))
        self.assertFalse(voice.is_azure_v1_voice("typecast:tc_abc"))

    def test_word_timestamps_become_script_aligned_subtitles(self):
        words = [
            {"text": "첫", "start": 0.0, "end": 0.2},
            {"text": "번째,", "start": 0.2, "end": 0.6},
            {"text": "스프레이.", "start": 0.7, "end": 1.4},
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            voice.requests, "post", return_value=_ok_response(words)
        ) as post, patch.object(voice, "AudioFileClip", _FakeClip):
            out = Path(tmp) / "audio.mp3"
            sub_maker = voice.typecast_tts("첫 번째, 스프레이.", "tc_abc", str(out))
            self.assertEqual(out.read_bytes(), b"fake-mp3")

            srt = Path(tmp) / "sub.srt"
            voice.create_subtitle(sub_maker, "첫 번째, 스프레이.", str(srt))
            content = srt.read_text(encoding="utf-8")

        self.assertEqual(post.call_count, 1)
        sent = post.call_args.kwargs
        self.assertEqual(sent["headers"]["X-API-KEY"], "test-key")
        self.assertEqual(sent["params"], {"granularity": "word"})
        self.assertEqual(sent["json"]["voice_id"], "tc_abc")
        self.assertIn("첫 번째", content)
        self.assertIn("00:00:00,700 --> 00:00:01,400", content)

    def test_error_response_is_never_resubmitted(self):
        for status in (400, 401, 402, 429, 500):
            with self.subTest(status=status):
                response = Mock(status_code=status, text="error")
                with patch.object(voice.requests, "post", return_value=response) as post:
                    self.assertIsNone(voice.typecast_tts("안녕.", "tc_abc", "unused.mp3"))
                self.assertEqual(post.call_count, 1, "paid synthesis was resubmitted")

    def test_connect_timeout_before_sending_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            voice.requests,
            "post",
            side_effect=[requests.ConnectTimeout("not connected"), _ok_response([])],
        ) as post, patch.object(voice, "AudioFileClip", _FakeClip):
            result = voice.typecast_tts("안녕.", "tc_abc", str(Path(tmp) / "a.mp3"))
        self.assertIsNotNone(result)
        self.assertEqual(post.call_count, 2)

    def test_missing_key_or_long_text_makes_no_request(self):
        with patch.object(voice.requests, "post") as post:
            with patch.object(voice, "get_typecast_api_key", return_value=""):
                self.assertIsNone(voice.typecast_tts("안녕.", "tc_abc", "unused.mp3"))
            self.assertIsNone(voice.typecast_tts("가" * 2001, "tc_abc", "unused.mp3"))
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
