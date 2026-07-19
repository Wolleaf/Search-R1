import json
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from search_r1.search.bm25_server import (BM25Retriever, _IndexedJsonlCorpus,
                                         _PyseriniSimpleSearcher,
                                         _parse_document)


class FakeHit:

    def __init__(self, docid, score):
        self.docid = docid
        self.score = score


class FakeDocument:

    def __init__(self, payload):
        self.payload = payload

    def raw(self):
        return None if self.payload is None else json.dumps(self.payload)


class FakeJavaDocument:

    def get(self, field):
        if field != "raw":
            raise AssertionError(field)
        return json.dumps({"contents": '"Java title"\nJava passage.'})


class FakeSearcher:

    def __init__(self):
        self.documents = {
            "1": FakeDocument({"contents": '"Title One"\nFirst passage.'}),
            "2": FakeDocument({"contents": '"Title Two"\nSecond passage.'}),
        }

    def search(self, query, topk):
        if query == "empty":
            return []
        return [FakeHit("1", 2.5), FakeHit("2", 1.25)][:topk]

    def doc(self, docid):
        return self.documents.get(docid)


class Bm25ServerTest(unittest.TestCase):

    def _write_corpus(self, root: Path, rows):
        corpus_path = root / "corpus.jsonl"
        offsets_path = root / "corpus.offsets"
        position = 0
        offsets = [0]
        with corpus_path.open("wb") as corpus:
            for index, contents in enumerate(rows):
                ending = b"\n" if index + 1 < len(rows) else b""
                line = json.dumps({
                    "id": str(index),
                    "contents": contents,
                }, ensure_ascii=False).encode() + ending
                corpus.write(line)
                position += len(line)
                offsets.append(position)
        offsets_path.write_bytes(b"".join(
            struct.pack("<Q", value) for value in offsets))
        return corpus_path, offsets_path

    def test_parse_document_preserves_contents(self):
        result = _parse_document(json.dumps({"contents": '"A title"\nBody text'}))
        self.assertEqual(result["title"], "A title")
        self.assertEqual(result["text"], "Body text")
        self.assertEqual(result["contents"], '"A title"\nBody text')

    def test_batch_search_matches_generation_contract(self):
        retriever = BM25Retriever("unused", topk=3, searcher=FakeSearcher())
        result = retriever.batch_search(["question", "empty"], return_scores=True)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]["document"]["title"], "Title One")
        self.assertEqual(result[0][0]["score"], 2.5)
        self.assertEqual(result[1], [])

    def test_blank_query_does_not_call_searcher(self):
        retriever = BM25Retriever("unused", searcher=FakeSearcher())
        self.assertEqual(retriever.search("   ", return_scores=True), [])

    def test_invalid_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contents"):
            _parse_document(json.dumps({"id": "missing"}))

    def test_external_corpus_supplies_documents_for_id_only_index(self):
        class IdOnlySearcher:

            def search(self, query, topk):
                return [FakeHit("0", 2.0), FakeHit("1", 1.0)][:topk]

            def doc(self, docid):
                return FakeDocument(None)

        with tempfile.TemporaryDirectory() as temporary:
            corpus_path, offsets_path = self._write_corpus(
                Path(temporary),
                ['"First"\nAlpha', '"Second"\nUnicode text'],
            )
            retriever = BM25Retriever(
                "unused",
                topk=2,
                searcher=IdOnlySearcher(),
                corpus_path=str(corpus_path),
                offsets_path=str(offsets_path),
            )
            result = retriever.search("question", return_scores=True)
            retriever.corpus.close()

        self.assertEqual(result[0]["document"]["title"], "First")
        self.assertEqual(result[1]["document"]["text"], "Unicode text")
        self.assertEqual(result[1]["score"], 1.0)

    def test_external_corpus_rejects_misaligned_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path, offsets_path = self._write_corpus(root, ["passage"])
            corpus_path.write_text('{"id":"9","contents":"passage"}')
            offsets_path.write_bytes(
                struct.pack("<QQ", 0, corpus_path.stat().st_size))
            corpus = _IndexedJsonlCorpus(str(corpus_path), str(offsets_path))
            with self.assertRaisesRegex(ValueError, "mismatched id"):
                corpus.get("0")
            corpus.close()

    def test_external_corpus_rejects_truncated_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = root / "corpus.jsonl"
            offsets_path = root / "corpus.offsets"
            corpus_path.write_text('{"id":"0","contents":"passage"}')
            offsets_path.write_bytes(struct.pack("<Q", 0))
            with self.assertRaisesRegex(ValueError, "invalid size"):
                _IndexedJsonlCorpus(str(corpus_path), str(offsets_path))

    def test_minimal_pyserini_adapter_avoids_dense_search_imports(self):
        class FakeJavaSearcher:

            def __init__(self, index_path):
                self.index_path = index_path

            def search(self, query, topk):
                return [(query, topk)]

            def doc(self, docid):
                return FakeJavaDocument() if docid == "known" else None

        pyserini = ModuleType("pyserini")
        pyclass = ModuleType("pyserini.pyclass")
        pyclass.autoclass = lambda name: (
            FakeJavaSearcher if name == "io.anserini.search.SimpleSearcher" else None)

        with mock.patch.dict(sys.modules, {
                "pyserini": pyserini,
                "pyserini.pyclass": pyclass,
        }):
            searcher = _PyseriniSimpleSearcher("index")

        self.assertEqual(searcher.search("query", 3), [("query", 3)])
        self.assertIn("Java passage", searcher.doc("known").raw())
        self.assertIsNone(searcher.doc("missing"))
        self.assertNotIn("pyserini.search.lucene", sys.modules)


if __name__ == "__main__":
    unittest.main()
