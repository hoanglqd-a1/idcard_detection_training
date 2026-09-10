import io
import tempfile
import unittest
import zipfile
from ftplib import error_perm
from pathlib import Path
from unittest.mock import Mock, patch

import download_dataset as dataset


class DownloadRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='download-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.filename = '01_alb_id.zip'
        self.zip_path = self.root / self.filename
        self.destination = self.root / '01_alb_id'
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('01_alb_id/images/CA/image.tif', b'image')
            archive.writestr('01_alb_id/ground_truth/CA/image.json', '{}')
        self.payload = buffer.getvalue()
        self.ftp = Mock()
        self.ftp.size.return_value = len(self.payload)
        self.ftp.retrbinary.side_effect = self.transfer
        factory = self.enterContext(patch.object(dataset, 'FTP'))
        factory.return_value.__enter__.return_value = self.ftp
        self.factory = factory
        self.enterContext(patch.object(dataset, 'tqdm'))
        self.enterContext(patch.object(dataset.time, 'sleep'))

    def transfer(self, command, callback, **kwargs):
        callback(self.payload)

    def test_download_validates_and_reuses_zip(self):
        result = dataset.download_zipfile(self.filename, self.root)
        self.assertEqual(result.read_bytes(), self.payload)
        self.factory.assert_called_once_with(dataset.HOST, timeout=30)
        self.assertFalse(self.zip_path.with_suffix('.zip.part').exists())
        dataset.download_zipfile(self.filename, self.root)
        self.assertEqual(self.ftp.retrbinary.call_count, 1)

    def test_size_is_optional(self):
        self.ftp.size.side_effect = error_perm('502 SIZE not implemented')
        dataset.download_zipfile(self.filename, self.root)
        self.assertEqual(self.zip_path.read_bytes(), self.payload)

    def test_interrupted_transfer_retries_from_beginning(self):
        calls = 0

        def interrupted(command, callback, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                callback(self.payload[:20])
                raise ConnectionResetError('Disconnected')
            callback(self.payload)

        self.ftp.retrbinary.side_effect = interrupted
        dataset.download_zipfile(self.filename, self.root)
        self.assertEqual(calls, 2)
        self.assertEqual(self.zip_path.read_bytes(), self.payload)

    def test_failed_attempts_do_not_publish_archive(self):
        self.ftp.retrbinary.side_effect = ConnectionResetError('Disconnected')
        with self.assertRaises(ConnectionResetError):
            dataset.download_zipfile(self.filename, self.root)
        self.assertEqual(self.ftp.retrbinary.call_count, 3)
        self.assertFalse(self.zip_path.exists())
        self.assertFalse(self.zip_path.with_suffix('.zip.part').exists())

    def test_invalid_cached_zip_is_replaced(self):
        self.zip_path.write_bytes(b'incomplete old download')
        dataset.download_zipfile(self.filename, self.root)
        self.assertEqual(self.zip_path.read_bytes(), self.payload)

    def test_short_or_invalid_download_is_not_published(self):
        for size in (len(self.payload), 3):
            with self.subTest(size=size):
                self.ftp.size.return_value = size
                self.ftp.retrbinary.side_effect = lambda command, callback, **kw: callback(b'bad')
                with self.assertRaises(zipfile.BadZipFile):
                    dataset.download_zipfile(self.filename, self.root, attempts=1)
                self.assertFalse(self.zip_path.exists())

    def test_incomplete_extraction_is_rebuilt_and_completed_cache_reused(self):
        self.zip_path.write_bytes(self.payload)
        self.destination.mkdir()
        (self.destination / 'incomplete.txt').write_text('old')
        result = dataset.prepare_archive(self.filename, self.root)
        self.assertTrue((result / '.extraction-complete').is_file())
        self.assertTrue((result / 'images/CA/image.tif').is_file())
        self.assertFalse((result / 'incomplete.txt').exists())
        with patch.object(dataset, 'download_zipfile') as download:
            self.assertEqual(dataset.prepare_archive(self.filename, self.root), result)
            download.assert_not_called()
        self.factory.assert_not_called()

    def test_interrupted_extraction_preserves_previous_directory(self):
        self.zip_path.write_bytes(self.payload)
        self.destination.mkdir()
        original = self.destination / 'original.txt'
        original.write_text('keep until extraction succeeds')
        extract = zipfile.ZipFile.extract
        calls = 0

        def interrupted(archive, member, path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('Extraction interrupted')
            return extract(archive, member, path)

        with patch.object(zipfile.ZipFile, 'extract', interrupted):
            with self.assertRaises(OSError):
                dataset.prepare_archive(self.filename, self.root)
        self.assertTrue(original.is_file())
        self.assertFalse((self.destination / '.extraction-complete').exists())
        self.assertEqual(set(self.root.iterdir()), {self.zip_path, self.destination})


if __name__ == '__main__':
    unittest.main()
