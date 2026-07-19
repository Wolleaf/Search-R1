"""Minimal CPU-only BM25 retrieval service for the AutoDL reproduction."""

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional


class _RawDocument:
    """Expose the small document API consumed by ``BM25Retriever``."""

    def __init__(self, document: Any):
        self.document = document

    def raw(self) -> str:
        return self.document.get("raw")


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


def _parse_document(raw_document: str) -> Dict[str, str]:
    payload = json.loads(raw_document)
    contents = payload.get("contents")
    if not isinstance(contents, str) or not contents.strip():
        raise ValueError("BM25 index document is missing a non-empty 'contents' field")
    title, _, text = contents.partition("\n")
    return {
        "title": title.strip('"'),
        "text": text,
        "contents": contents,
    }


class BM25Retriever:
    """Thin wrapper around Pyserini that does not import dense-retrieval packages."""

    def __init__(self, index_path: str, topk: int = 3, searcher: Optional[Any] = None):
        if topk <= 0:
            raise ValueError("topk must be positive")
        if searcher is None:
            searcher = _PyseriniSimpleSearcher(index_path)
        self.searcher = searcher
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
            item: Dict[str, Any] = {"document": _parse_document(document.raw())}
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
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    retriever = BM25Retriever(args.index_path, topk=args.topk)
    app = create_app(retriever)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
