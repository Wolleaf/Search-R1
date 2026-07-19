import hashlib
from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.autodl.build_corpus_offsets import (MEMBER_NAME, prepare,
                                                 sha256_file, verify)
from search_r1.search.bm25_server import _IndexedJsonlCorpus


class CorpusOffsetsTest(unittest.TestCase):

    def _write_archive(self, root: Path, rows):
        payload = b"".join(rows)
        source = root / "wiki-18.jsonl.gz"
        with tarfile.open(source, "w:gz") as archive:
            member = tarfile.TarInfo(MEMBER_NAME)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
        return source, payload

    def test_prepare_and_verify_offset_corpus(self):
        rows = [
            b'{"id":"0","contents":"First"}\n',
            '{"id":"1","contents":"Unicode"}'.encode(),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, payload = self._write_archive(root, rows)
            output = root / "corpus"
            source_digest = sha256_file(source)

            manifest = prepare(source, output, "revision", source_digest,
                               source.stat().st_size, len(payload))
            verified = verify(output, "revision", source_digest,
                              source.stat().st_size, len(payload))
            reused = prepare(source, output, "revision", source_digest,
                             source.stat().st_size, len(payload))
            corpus = _IndexedJsonlCorpus(
                str(output / "wiki-18.jsonl"),
                str(output / "wiki-18.offsets.u64"),
            )
            second = corpus.get("1")
            corpus.close()

        self.assertEqual(manifest, verified)
        self.assertEqual(manifest, reused)
        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(second["contents"], "Unicode")
        self.assertEqual(manifest["artifacts"]["wiki-18.jsonl"]["sha256"],
                         hashlib.sha256(payload).hexdigest())

    def test_prepare_rejects_nonsequential_ids(self):
        rows = [b'{"id":"4","contents":"Wrong id"}\n']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, payload = self._write_archive(root, rows)
            output = root / "corpus"
            with self.assertRaisesRegex(ValueError, "mismatched id"):
                prepare(source, output, "revision", sha256_file(source),
                        source.stat().st_size, len(payload))
            self.assertFalse((output / "manifest.json").exists())

    def test_prepare_refuses_interrupted_staging_directory(self):
        rows = [b'{"id":"0","contents":"passage"}\n']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, payload = self._write_archive(root, rows)
            output = root / "corpus"
            (root / ".corpus.building").mkdir()
            with self.assertRaisesRegex(ValueError, "interrupted corpus build"):
                prepare(source, output, "revision", sha256_file(source),
                        source.stat().st_size, len(payload))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
