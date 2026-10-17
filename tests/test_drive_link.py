#!/usr/bin/env python3
"""Ссылки на Google Drive: временная пометка drivefs не должна превращаться в ссылку.

У только что созданной на смонтированном диске папки расширенный атрибут `user.drive.id`
содержит временную пометку («local-105»). Ссылка по ней не открывается — пользователь жмёт
и не видит ничего. Такие пометки считаем отсутствием идентификатора и отдаём поиск по имени.
"""
import importlib.util
import os
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("qwen_tts_book", ROOT / "colab" / "cli" / "qwen_tts_book.py")
book = importlib.util.module_from_spec(spec)

# На машине конвейера Drive подключён; для проверки ссылок это не нужно — модуль читает путь
# только в pack(), а нам нужен один drive_link().
spec.loader.exec_module(book)


class DriveLink(unittest.TestCase):
    def setUp(self):
        # Ссылку на папку функция строит только для существующего каталога (is_dir),
        # поэтому проверяем на настоящих файлах и папках с именем книги.
        self.work = pathlib.Path(tempfile.mkdtemp())
        self.folder = self.work / "Книга"
        self.folder.mkdir()
        self.file = self.folder / "Книга.m4b"
        self.file.write_bytes(b"fake")

    def with_id(self, path, value, kind):
        """Подставляет значение атрибута вместо настоящего getxattr."""
        original = getattr(os, "getxattr", None)
        os.getxattr = lambda *_args, **_kwargs: value
        try:
            return book.drive_link(path)
        finally:
            if original is None:
                del os.getxattr
            else:
                os.getxattr = original

    def test_real_id_gives_folder_and_file_links(self):
        real = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456"
        self.assertEqual(self.with_id(self.folder, real.encode(), "dir"),
                         f"https://drive.google.com/drive/folders/{real}")
        self.assertEqual(self.with_id(self.file, real.encode(), "file"),
                         f"https://drive.google.com/file/d/{real}/view")

    def test_local_placeholder_is_not_a_link(self):
        link = self.with_id(self.folder, b"local-105", "dir")
        self.assertNotIn("local-105", link)
        self.assertIn("drive.google.com/drive/search?q=", link)
        self.assertIn("%D0%9A", link)                 # имя книги в ссылке

    def test_short_value_is_not_an_id(self):
        self.assertIn("/drive/search?q=", self.with_id(self.folder, b"1AbC", "dir"))

    def test_no_attribute_gives_search(self):
        def boom(*_args, **_kwargs):
            raise OSError("no such attribute")

        original = getattr(os, "getxattr", None)
        os.getxattr = boom
        try:
            self.assertIn("/drive/search?q=", book.drive_link(self.folder))
        finally:
            if original is None:
                del os.getxattr
            else:
                os.getxattr = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
