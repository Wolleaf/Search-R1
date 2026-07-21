"""Minimal CPU-only BM25 retrieval service for the AutoDL reproduction."""

import argparse
import json
import mmap
from pathlib import Path
import struct
from typing import Any, Dict, Iterable, List, Optional


class _RawDocument:
    """Expose the small document API consumed by ``BM25Retriever``."""

    def __init__(self, document: Any):
        self.document = document

    def raw(self) -> Optional[str]:
        return self.document.get("raw")


class _IndexedJsonlCorpus:
    """Read corpus rows by Lucene docid without loading Wikipedia into RAM."""

    _OFFSET_WIDTH = 8

    def __init__(self, corpus_path: str, offsets_path: str):
        self.corpus_path = Path(corpus_path)
        self.offsets_path = Path(offsets_path)
        self._corpus_file = None
        self._corpus = None
        self._offsets_file = None
        self._offsets = None
        try:
            self._corpus_file = self.corpus_path.open("rb")
            self._corpus = mmap.mmap(
                self._corpus_file.fileno(), 0, access=mmap.ACCESS_READ)
            self._offsets_file = self.offsets_path.open("rb")
            self._offsets = mmap.mmap(
                self._offsets_file.fileno(), 0, access=mmap.ACCESS_READ)
            if (len(self._offsets) < 2 * self._OFFSET_WIDTH
                    or len(self._offsets) % self._OFFSET_WIDTH):
                raise ValueError("corpus offset index has an invalid size")
            self.rows = len(self._offsets) // self._OFFSET_WIDTH - 1
            if self._offset(self.rows) != len(self._corpus):
                raise ValueError("corpus offset index does not match the JSONL file")
        except BaseException:
            self.close()
            raise

    def _offset(self, index: int) -> int:
        return struct.unpack_from("<Q", self._offsets,
                                  index * self._OFFSET_WIDTH)[0]

    def get(self, docid: str) -> Dict[str, str]:
        try:
            index = int(docid)
        except (TypeError, ValueError) as error:
            raise ValueError(f"BM25 returned a non-integer document id: {docid!r}") from error
        if index < 0 or index >= self.rows:
            raise ValueError(f"BM25 document id is outside the corpus: {docid!r}")

        start = self._offset(index)
        end = self._offset(index + 1)
        if end < start:
            raise ValueError("corpus offset index is not monotonic")
        payload = json.loads(self._corpus[start:end])
        if str(payload.get("id")) != str(index):
            raise ValueError(
                f"corpus row {index} has mismatched id {payload.get('id')!r}")
        return _normalize_document(payload)

    def close(self) -> None:
        offsets = getattr(self, "_offsets", None)
        if offsets is not None:
            offsets.close()
            self._offsets = None
        offsets_file = getattr(self, "_offsets_file", None)
        if offsets_file is not None:
            offsets_file.close()
            self._offsets_file = None
        corpus = getattr(self, "_corpus", None)
        if corpus is not None:
            corpus.close()
            self._corpus = None
        corpus_file = getattr(self, "_corpus_file", None)
        if corpus_file is not None:
            corpus_file.close()
            self._corpus_file = None

    def __del__(self) -> None:
        self.close()


class _PyseriniSimpleSearcher:
    """Load Anserini directly without Pyserini's optional dense imports."""

    def __init__(self, index_path: str):
        from pyserini.pyclass import autoclass

        simple_searcher = autoclass("io.anserini.search.SimpleSearcher")
        self.searcher = simple_searcher(index_path)

    def search(self, query: str, topk: int) -> Any:
        return self.searcher.search(query, topk)

    def doc(self, docid: str) -> Optional[_RawDocument]:
        document = self.searcher.doc(docid)
        return None if document is None else _RawDocument(document)


def _normalize_document(payload: Dict[str, Any]) -> Dict[str, str]:
    contents = payload.get("contents")
    if not isinstance(contents, str) or not contents.strip():
        raise ValueError("BM25 index document is missing a non-empty 'contents' field")
    title, _, text = contents.partition("\n")
    return {
        "title": title.strip('"'),
        "text": text,
        "contents": contents,
    }


def _parse_document(raw_document: str) -> Dict[str, str]:
    return _normalize_document(json.loads(raw_document))


class BM25Retriever:
    """Thin wrapper around Pyserini that does not import dense-retrieval packages."""

    def __init__(self,
                 index_path: str,
                 topk: int = 3,
                 searcher: Optional[Any] = None,
                 corpus: Optional[Any] = None,
                 corpus_path: Optional[str] = None,
                 offsets_path: Optional[str] = None):
        if topk <= 0:
            raise ValueError("topk must be positive")
        if searcher is None:
            searcher = _PyseriniSimpleSearcher(index_path)
        if corpus is not None and (corpus_path is not None or offsets_path is not None):
            raise ValueError("pass either a corpus object or corpus paths, not both")
        if (corpus_path is None) != (offsets_path is None):
            raise ValueError("corpus_path and offsets_path must be provided together")
        if corpus is None and corpus_path is not None and offsets_path is not None:
            corpus = _IndexedJsonlCorpus(corpus_path, offsets_path)
        self.searcher = searcher
        self.corpus = corpus
        self.topk = topk

    def search(self, query: str, topk: Optional[int] = None,
               return_scores: bool = False) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        limit = self.topk if topk is None else topk
        if limit <= 0:
            raise ValueError("topk must be positive")

        output = []
        for hit in self.searcher.search(query, limit):
            document = self.searcher.doc(hit.docid)
            if document is None:
                raise ValueError(f"BM25 index returned missing document {hit.docid}")
            raw_document = document.raw()
            if raw_document is None:
                if self.corpus is None:
                    raise ValueError(
                        "BM25 index stores only ids, but no external corpus was configured")
                parsed_document = self.corpus.get(hit.docid)
            else:
                parsed_document = _parse_document(raw_document)
            item: Dict[str, Any] = {
                "document": parsed_document,
                "document_id": str(hit.docid),
            }
            if return_scores:
                item["score"] = float(hit.score)
            output.append(item)
        return output

    def batch_search(self, queries: Iterable[str], topk: Optional[int] = None,
                     return_scores: bool = False) -> List[List[Dict[str, Any]]]:
        return [self.search(query, topk, return_scores) for query in queries]


def create_app(retriever: BM25Retriever):
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class QueryRequest(BaseModel):
        queries: List[str]
        topk: Optional[int] = Field(default=None, gt=0)
        return_scores: bool = False

    app = FastAPI(title="Search-R1 BM25 Retriever")

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        results = retriever.batch_search(
            request.queries,
            topk=request.topk,
            return_scores=request.return_scores,
        )
        return {"result": results}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--offsets-path", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    retriever = BM25Retriever(
        args.index_path,
        topk=args.topk,
        corpus_path=args.corpus_path,
        offsets_path=args.offsets_path,
    )
    app = create_app(retriever)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
