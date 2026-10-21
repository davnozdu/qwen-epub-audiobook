"""Многоголосая книга: разметка ролей (cast_book.py) и пакет голосов (voice_pack.py) — без сети и GPU."""
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "colab" / "cli"))
sys.path.insert(0, str(ROOT / "colab"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cast_book  # noqa: E402
import voice_pack  # noqa: E402


def make_book(path: Path) -> Path:
    paragraphs = [
        "Утро было тихим, и старый дом спал над рекой.",
        "— Вставай, Петя, — сказала мать. — Пора в дорогу.",
        "— Не хочу, — пробормотал Петя и отвернулся к стене.",
        "Слово «дорога» звучало для него скучно.",
        "— Кто там? — крикнул кто-то из сада.",
    ]
    body = "".join(f"<p>{text}</p>" for text in paragraphs)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OPS/c.opf"/></rootfiles></container>')
        z.writestr("OPS/c.opf",
                   '<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<metadata><dc:title>Утро</dc:title><dc:creator>Тест</dc:creator></metadata>'
                   '<manifest><item id="a" href="a.xhtml" media-type="application/xhtml+xml"/></manifest>'
                   '<spine><itemref idref="a"/></spine></package>')
        z.writestr("OPS/a.xhtml", f"<html><body><h1>Глава</h1>{body}</body></html>")
    return path


def fake_llm(body):
    """Поддельная LLM: отвечает по виду запроса так, как ответила бы настоящая."""
    prompt = body["messages"][0]["content"]
    if prompt.startswith("Ты режиссёр русской аудиокниги. Книга"):
        answer = {"characters": [
            {"id": "narrator", "name": "Рассказчик", "gender": "male", "age": "adult", "aliases": [],
             "evidence": "текст", "voice_description": "calm"},
            {"id": "мать", "name": "Мать", "gender": "female", "age": "adult", "aliases": [], "evidence": "p00003",
             "voice_description": "warm"},
            {"id": "petya", "name": "Петя", "gender": "male", "age": "boy", "aliases": ["Петька"],
             "evidence": "p00004", "voice_description": "young"}]}
    elif prompt.startswith("Разметь русскую книгу"):
        payload = json.loads(prompt[prompt.index("{\"cast\""):])
        rows = []
        for item in payload["target"]:
            speech = []
            for fragment in item["fragments"]:
                text = fragment["t"]
                if text.startswith("— Вставай") or text.startswith("— Пора"):
                    speech.append({"n": fragment["n"], "speaker": "mat", "manner": "calm"})
                elif text.startswith("— Не хочу"):
                    speech.append({"n": fragment["n"], "speaker": "Petya", "manner": "muttering"})
                elif text.startswith("— Кто там"):
                    speech.append({"n": fragment["n"], "speaker": None, "manner": "shouting"})
            rows.append({"id": item["id"], "speech": speech, "confidence": "high"})
        answer = {"paragraphs": rows}
    elif prompt.startswith("Определи говорящего"):
        payload = json.loads(prompt[prompt.index("{\"voices\""):])
        answer = {"assignments": [{"id": s["id"], "speaker": None, "gender": "female"} for s in payload["segments"]]}
    elif prompt.startswith("Ты режиссёр русской аудиокниги. Текст"):
        payload = json.loads(prompt[prompt.index("{\"book\""):])
        cid = payload["character"]["id"]
        answer = {"id": cid, "age": "взрослый", "personality": "добрый", "habits": "вздыхает",
                  "speaking_style": "мягко", "voice_description": f"Warm voice of {cid}", "certainty": "medium",
                  "assumptions": [], "evidence": [{"kind": "book", "reference": "p00003", "note": "реплика"},
                                                 {"kind": "web", "reference": "http://x", "note": "лишнее"}]}
    elif prompt.startswith("You write the voice description"):
        payload = json.loads(re.search(r"\{\"voice_gender\".*?\}", prompt).group(0))
        tag = payload["voice_description"].split()[-1]
        answer = {"voice": f"Warm resonant voice for {tag}, adult, mid pitch, smooth timbre",
                  "delivery": "Lively melodic delivery with clear emphasis on key words and short pauses"}
    else:
        raise AssertionError("неожиданный запрос: " + prompt[:80])
    return {"done_reason": "stop", "message": {"content": "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"}}


class FragmentsTest(unittest.TestCase):
    def test_dialogue_and_author_words(self):
        text = "— Ну что же, — сказал он, — пойдём. «Это не речь» и «Да ты что! Правда?» — спросила она."
        parts = [text[a:b] for a, b in cast_book.fragments(text)]
        self.assertEqual(parts, ["— Ну что же,", "— сказал он,", "— пойдём. «Это не речь» и",
                                 "«Да ты что! Правда?»", "— спросила она."])

    def test_fragments_cover_all_text(self):
        text = "Он сказал: «Иди сюда». Потом — тишина, — и снова: «Ну?»"
        covered = "".join(text[a:b] for a, b in cast_book.fragments(text))
        self.assertEqual(re.sub(r"\s", "", covered), re.sub(r"\s", "", text))

    def test_cast_id_folds_cyrillic_and_diacritics(self):
        self.assertEqual(cast_book.cast_id("Игорь Петрович"), "igor_petrovich")
        self.assertEqual(cast_book.cast_id("dühring"), "duhring")
        self.assertEqual(cast_book.cast_id("12 стульев"), "c_12_stulev")


class SpeakersTest(unittest.TestCase):
    def test_spelling_variant_unknown_speaker_and_merge(self):
        chunk = [{"id": "p00001", "text": "— Да, — сказал он. — Нет."}]
        parts = [cast_book.fragments(chunk[0]["text"])]
        answer = {"paragraphs": [{"id": "p00001", "speech": [{"n": 1, "speaker": "Ivan", "manner": "angry"},
                                                             {"n": 3, "speaker": "stranger"}]}]}
        spans = cast_book.validate_speakers(answer, chunk, parts, {"narrator", "ivan"})[0]["spans"]
        self.assertEqual([s["speaker"] for s in spans], ["ivan", "narrator", None])
        self.assertEqual(spans[0]["manner"], "angry")

    def test_wrong_paragraph_order_is_rejected(self):
        chunk = [{"id": "p00001", "text": "Текст."}]
        with self.assertRaises(cast_book.LLMError):
            cast_book.validate_speakers({"paragraphs": [{"id": "p00002", "speech": []}]}, chunk, [[(0, 6)]], {"narrator"})

    def test_merge_cast_keeps_identity(self):
        cast = cast_book.merge_cast([], [{"id": "ivan", "name": "Иван", "gender": "unknown"}])
        cast = cast_book.merge_cast(cast, [{"id": "ivan", "name": "Ванька", "gender": "male"}])
        self.assertEqual(cast[0]["name"], "Иван")
        self.assertEqual(cast[0]["aliases"], ["Ванька"])
        self.assertEqual(cast[0]["gender"], "male")


class SampleTextTest(unittest.TestCase):
    def test_sample_is_short_and_ends_with_a_sentence(self):
        long_line = ("«Высвободить бы только руки, я сбросил бы петлю и прыгнул в воду. Если глубоко нырнуть, "
                     "пули меня не достанут, я бы доплыл до берега, скрылся в лесу и пробрался домой.")
        source = {"paragraphs": [{"id": "p00001", "text": long_line}]}
        labels = [{"id": "p00001", "spans": [{"start": 0, "end": len(long_line), "speaker": "hero"}]}]
        text = cast_book.excerpt(source, labels, "hero")
        self.assertLessEqual(len(text), cast_book.SAMPLE_MAX)
        self.assertEqual(text, "Высвободить бы только руки, я сбросил бы петлю и прыгнул в воду.")

    def test_long_designed_sample_is_redone(self):
        import numpy as np
        seconds = voice_pack.MAX_SAMPLE_SECONDS + 3
        audio = (0.3 * np.sin(2 * np.pi * 200 * np.arange(int(voice_pack.RATE * seconds)) / voice_pack.RATE))
        self.assertIn("длиннее", voice_pack.plausible(audio, "а" * int(seconds * 12)))


class ProviderTest(unittest.TestCase):
    def test_deepseek_request_and_reply(self):
        seen = []

        def transport(body):
            seen.append(body)
            return {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": 1}'}}]}

        llm = cast_book.LLM("k", "deepseek-flash", transport=transport, provider="deepseek")
        self.assertEqual(llm.call("json please"), {"ok": 1})
        self.assertEqual(seen[0]["response_format"], {"type": "json_object"})
        self.assertEqual(seen[0]["thinking"], {"type": "disabled"})
        self.assertEqual(llm.endpoint, cast_book.DEEPSEEK_ENDPOINT)

    def test_truncated_reply_is_retried(self):
        replies = [{"choices": [{"finish_reason": "length", "message": {"content": "{"}}]},
                   {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": 2}'}}]}]
        llm = cast_book.LLM("k", transport=lambda body: replies.pop(0), provider="deepseek")
        self.assertEqual(llm.call("x"), {"ok": 2})


class AnalyzeTest(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp())
        self.book = make_book(self.work / "book.epub")
        self.cast_path = cast_book.analyze(self.book, self.work / "cast", "key", transport=fake_llm,
                                           sleep=lambda seconds: None)
        self.cast = json.loads(self.cast_path.read_text(encoding="utf-8"))

    def test_cast_roles_and_reserve_voice(self):
        voices = {v["id"]: v for v in self.cast["voices"]}
        self.assertTrue(voices["narrator"]["fixed"])
        self.assertIn("mat", voices)                     # ID кириллицей переведён в латиницу
        self.assertIn("other_voices_female", voices)    # «Кто там?» — неизвестная, по грамматике женщина
        self.assertEqual(voices["petya"]["lines"], 1)
        for vid in ("mat", "petya"):
            self.assertTrue(voices[vid]["instruct"].endswith(cast_book.READING_RULES))
            self.assertEqual(voices[vid]["profile"]["evidence"][0]["kind"], "book")   # веб-ссылки отброшены
        self.assertEqual(voices["petya"]["sample_text"], "Это мой голос в этой книге. Сейчас я расскажу свою часть истории.")

    def test_cache_makes_second_run_free(self):
        calls = []
        cast_book.analyze(self.book, self.work / "cast", "key",
                          transport=lambda body: calls.append(1) or fake_llm(body), sleep=lambda seconds: None)
        self.assertEqual(calls, [])

    def test_manifest_from_cast_passes_pipeline_checks(self):
        manifest = json.loads(cast_book.build_manifest(self.book, self.work / "project", self.cast_path)
                              .read_text(encoding="utf-8"))
        speakers = {s["speaker"] for s in manifest["segments"]}
        self.assertEqual(speakers, {"narrator", "mat", "petya", "other_voices_female"})
        mother = [s["text"] for s in manifest["segments"] if s["speaker"] == "mat"]
        self.assertTrue(any(text.startswith("— Вставай") for text in mother))
        self.assertTrue(all("сказала" not in text for text in mother))   # ремарка автора — рассказчику

    def test_part_of_book_finds_its_labels_by_text(self):
        part = make_book(self.work / "part.epub")
        with zipfile.ZipFile(part, "a") as z:                  # та же книга, другое имя файла
            pass
        manifest = json.loads(cast_book.build_manifest(part, self.work / "part", self.cast_path)
                              .read_text(encoding="utf-8"))
        self.assertIn("petya", {s["speaker"] for s in manifest["segments"]})


class VoicePackTest(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp())
        self.cast = {"format": "qwen-voice-cast", "voices": [
            {"id": "narrator", "name": "Рассказчик", "fixed": True},
            {"id": "a", "name": "Анна", "gender": "female", "seed": 17001, "lines": 5, "sample_text": "Привет, как дела у тебя сегодня?",
             "tts_voice": "Bright female voice", "tts_delivery": "Lively delivery", "instruct": "A"},
            {"id": "b", "name": "Белла", "gender": "female", "seed": 17002, "lines": 3, "sample_text": "Привет, как дела у тебя сегодня?",
             "tts_voice": "Soft female voice", "tts_delivery": "Calm delivery", "instruct": "B"}],
            "paragraphs": []}
        self.cast_path = self.work / "cast.json"
        self.cast_path.write_text(json.dumps(self.cast, ensure_ascii=False), encoding="utf-8")

    def tone(self, frequency, seconds=2.5):
        import numpy as np
        t = np.arange(int(voice_pack.RATE * seconds)) / voice_pack.RATE
        wave = 0.3 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.sin(4 * np.pi * frequency * t)
        return wave.astype("float32")

    def test_measure_pitch(self):
        sound = voice_pack.measure(self.tone(200), "Привет, как дела у тебя сегодня?")
        self.assertAlmostEqual(sound["pitch"], 200, delta=5)

    def test_twin_is_redesigned_and_pack_round_trips(self):
        calls = []

        def generate(text, instruct, seed):
            calls.append(instruct)
            return self.tone(260 if "higher" in instruct else 200)    # «Белла» сначала звучит как «Анна»

        pack = voice_pack.design(self.cast_path, self.work / "pack", generate=generate)
        meta = json.loads((pack / "voices" / "b.json").read_text(encoding="utf-8"))
        self.assertIn("higher", meta["instruct"])
        self.assertIn("Анна", meta["note"])
        archive = voice_pack.zip_pack(pack, self.work / "voices.zip")
        opened = voice_pack.open_pack(archive, self.work / "opened")
        self.assertTrue((opened / "voices" / "a.wav").is_file())
        before = len(calls)
        voice_pack.design(self.cast_path, self.work / "pack", generate=generate)
        self.assertEqual(len(calls), before)                          # готовые образцы не пересоздаются

    def test_silent_sample_is_retried_with_another_seed(self):
        import numpy as np
        seeds = []

        def generate(text, instruct, seed):
            seeds.append(seed)
            return np.zeros(24000, dtype="float32") if len(seeds) == 1 else self.tone(200 + 70 * len(seeds))

        voice_pack.design(self.cast_path, self.work / "pack2", generate=generate)
        self.assertEqual(seeds[:2], [17001, 18001])

    def test_broken_pack_is_rejected(self):
        pack = voice_pack.design(self.cast_path, self.work / "pack3",
                                 generate=lambda text, instruct, seed: self.tone(150 + seed % 7 * 40))
        (pack / "voices" / "a.wav").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            voice_pack.open_pack(pack, self.work / "unused")


class WrapperTest(unittest.TestCase):
    def test_job_arguments_for_both_machines(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_book", ROOT / "colab" / "cli" / "run-book.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        job = Path(tempfile.mkdtemp()) / "job.py"
        job.write_text("print('job')\n", encoding="utf-8")
        shrink = {"max_hours": 7, "dir": "x", "off": True}
        first = module.build_job_file("/content/b.epub", "", "", False, "d", job, shrink, cast=True).read_text()
        self.assertIn('"--cast"', first)
        second = module.build_job_file("/content/b.epub", "", "", False, "d", job, shrink, {"part": "2"},
                                       voice_pack="/content/b-voices.zip").read_text()
        self.assertIn('"--voice-pack", "/content/b-voices.zip"', second)
        self.assertIn('"--raw"', second)


if __name__ == "__main__":
    unittest.main()


class CastStageTest(unittest.TestCase):
    def test_cast_stage_runs_analysis_and_packs_voices(self):
        """Стадия первой машины целиком (сбой на GPU был в заголовке стадии): разметка, голоса, архив."""
        import argparse
        import os
        import colab_run_all
        work = Path(tempfile.mkdtemp())
        book = make_book(work / "book.epub")
        real_analyze, real_design, real_model = cast_book.analyze, voice_pack.design, colab_run_all.stage_model
        cast_book.analyze = lambda book_path, out, key, model, provider="ollama": real_analyze(
            book_path, out, key, model, transport=fake_llm, sleep=lambda s: None, provider=provider)
        designed = []
        voice_pack.design = lambda cast_path, pack, design_model=None: (designed.append(design_model),
                                                                         real_design(cast_path, pack, generate=lambda
                                                                         text, instruct, seed: VoicePackTest.tone(None, 150 + seed % 97)))[1]
        colab_run_all.stage_model = lambda args, repo=None, name=None, model_dir=None: str(model_dir)
        os.environ["LLM_API_KEY"] = "k"
        try:
            args = argparse.Namespace(book=str(book), out=str(work / "out"), rebuild=False,
                                      cast_zip=str(work / "out" / "book-voices.zip"))
            target = colab_run_all.stage_cast(args)
        finally:
            cast_book.analyze, voice_pack.design, colab_run_all.stage_model = real_analyze, real_design, real_model
            os.environ.pop("LLM_API_KEY", None)
        self.assertTrue(Path(target).is_file())
        self.assertTrue(designed and "VoiceDesign" in designed[0])
        names = zipfile.ZipFile(target).namelist()
        self.assertIn("cast.json", names)
        self.assertIn("voices/petya.wav", names)


class EndToEndRolesTest(unittest.TestCase):
    """Разметка → манифест → подготовка → стадия voices с пакетом → синтез: каждую реплику модель
    получает с образцом ИМЕННО её героя, слова автора — с образцом рассказчика."""

    def test_each_line_is_cloned_from_its_own_speaker(self):
        import argparse
        import numpy as np
        import soundfile as sf
        import audiobook_epub
        import colab_run_all
        from test_synthesis_groups import FakeModel, fake_mlx, SR
        import synthesize_qwen_mlx as q

        work = Path(tempfile.mkdtemp())
        book = make_book(work / "book.epub")
        cast_path = cast_book.analyze(book, work / "cast", "key", transport=fake_llm, sleep=lambda s: None)
        cast = json.loads(cast_path.read_text(encoding="utf-8"))
        texts = {}
        for number, voice in enumerate(v for v in cast["voices"] if not v.get("fixed")):
            texts[voice["id"]] = f"Образец героя {voice['id']}."
            voice["sample_text"] = texts[voice["id"]]
        cast_path.write_text(json.dumps(cast, ensure_ascii=False), encoding="utf-8")
        tones = iter(range(150, 400, 60))
        pack = voice_pack.design(cast_path, work / "out" / "voice-pack", generate=lambda text, instruct, seed: (
            0.3 * np.sin(2 * np.pi * next(tones) * np.arange(SR * 2) / SR)).astype("float32"))

        project = work / "out" / "project"
        cast_book.build_manifest(book, project, pack / "cast.json")
        audiobook_epub.prepare(str(project / "manifest.json"), str(project / "source"), highlight="sentence",
                               start="p00001")
        narrator = work / "narrator.wav"
        sf.write(narrator, (0.2 * np.sin(np.arange(SR * 3) / SR * 2 * np.pi * 120)).astype("float32"), SR)
        (work / "narrator.json").write_text(json.dumps({"text": "Голос рассказчика."}, ensure_ascii=False))
        args = argparse.Namespace(out=str(work / "out"), voice=str(narrator), voice_pack=str(pack))
        library = colab_run_all.stage_voices(args, project / "source")

        class Recorder(FakeModel):
            refs = []

            def generate(self, text, **options):
                self.refs.append((text, options.get("ref_text")))
                yield from FakeModel.generate(self, text)

        model = Recorder()
        run = argparse.Namespace(manifest=str(project / "source/manifest.json"), model=str(work), out=str(work / "audio"),
                                 seconds=300, whole_book=True, start_paragraph="p00001", temperature=0.9,
                                 max_tokens=1024, stress_marks="keep", plan=False, mode="clone",
                                 rab_generation=False, voice_library=str(library), design_model=None,
                                 reference_source=None, import_only=False, ephemeral_voices=False)
        with fake_mlx(model):
            q.render(run)
        prepared = json.loads((project / "source/manifest.json").read_text(encoding="utf-8"))
        expected = dict(texts, narrator="Голос рассказчика.")
        checked = set()
        for text, ref_text in model.refs[1:]:                     # первый вызов — прогрев
            # Знаки не сравниваем: конец вызова модели закрывается точкой (closed_phrase).
            bare = lambda value: re.sub(r"[́\s.,;:!?…]", "", value)
            owners = {s["speaker"] for s in prepared["segments"] if bare(s["text"])[:12] in bare(text)}
            self.assertEqual(len(owners), 1, (text, owners))
            owner = owners.pop()
            self.assertEqual(ref_text, expected[owner], (text, owner))
            checked.add(owner)
        self.assertTrue({"narrator", "mat", "petya", "other_voices_female"} <= checked, checked)
