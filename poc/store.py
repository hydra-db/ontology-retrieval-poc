import hashlib
import json
import math
import re
import sqlite3
from collections import Counter


KINDS = {"Person", "Issue", "Comment", "CommentReply", "PullRequest", "Meeting", "TranscriptTurn", "Publication", "Author"}
RELATIONS = {"AUTHORED_BY", "PART_OF", "REPLY_TO", "HAS_AUTHOR", "HAS_TURN", "SPOKEN_BY"}
EDGE_TYPES = {
    "AUTHORED_BY": ({"Issue", "Comment", "CommentReply", "PullRequest"}, {"Person"}),
    "PART_OF": ({"Comment", "CommentReply"}, {"Issue", "PullRequest"}),
    "REPLY_TO": ({"Comment", "CommentReply"}, {"Comment", "CommentReply"}),
    "HAS_AUTHOR": ({"Publication"}, {"Author"}),
    "HAS_TURN": ({"Meeting"}, {"TranscriptTurn"}),
    "SPOKEN_BY": ({"TranscriptTurn"}, {"Person"}),
}


def words(text):
    return re.findall(r"[\w-]+", text.lower())


class Lexical:
    name = "lexical-cosine-v1"
    is_vector_model = False

    def encode(self, text):
        return dict(Counter(words(text)))

    def similarity(self, a, b):
        dot = sum(v * b.get(k, 0) for k, v in a.items())
        norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
        return dot / norm if norm else 0.0


class SentenceEncoder:
    is_vector_model = True

    def __init__(self, path):
        from pathlib import Path
        from sentence_transformers import SentenceTransformer

        root = Path(path).resolve()
        manifest = json.loads((root / "poc-model.json").read_text())
        self.name = manifest["model"] + "@" + manifest["revision"]
        self.model = SentenceTransformer(str(root), local_files_only=True, device="cpu")

    def encode(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def similarity(self, a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))


class CrossEncoderReranker:
    def __init__(self, path):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(path, local_files_only=True, device="cpu")

    def score(self, query, texts):
        return self.model.predict([(query, text) for text in texts]).tolist()


class Store:
    def __init__(self, path=":memory:", encoder=None):
        self.encoder = encoder or Lexical()
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
          tenant TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL,
          title TEXT NOT NULL, text TEXT NOT NULL, url TEXT NOT NULL,
          metadata TEXT NOT NULL, PRIMARY KEY(tenant,id));
        CREATE TABLE IF NOT EXISTS edges (
          tenant TEXT NOT NULL, src TEXT NOT NULL, relation TEXT NOT NULL, dst TEXT NOT NULL,
          PRIMARY KEY(tenant,src,relation,dst),
          FOREIGN KEY(tenant,src) REFERENCES nodes(tenant,id) ON DELETE CASCADE,
          FOREIGN KEY(tenant,dst) REFERENCES nodes(tenant,id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS chunks (
          tenant TEXT NOT NULL, id TEXT NOT NULL, node_id TEXT NOT NULL,
          start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL,
          embedding TEXT NOT NULL, model TEXT NOT NULL,
          PRIMARY KEY(tenant,id), FOREIGN KEY(tenant,node_id) REFERENCES nodes(tenant,id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS chunk_node ON chunks(tenant,node_id);
        """)

    def ingest(self, tenant, nodes, edges):
        if not tenant or len(nodes) > 10000:
            raise ValueError("tenant required; bounded batch of at most 10000 nodes")
        with self.db:
            for node in nodes:
                if node["kind"] not in KINDS or not node["id"] or len(node["text"]) > 2_000_000:
                    raise ValueError("invalid typed node")
                existing = self.node(tenant, node["id"])
                if existing and existing["kind"] != node["kind"]:
                    raise ValueError("stable node type cannot change")
                metadata = dict(node.get("metadata", {}))
                if node.get("author"):
                    metadata["author"] = node["author"]
                self.db.execute("""INSERT INTO nodes VALUES (?,?,?,?,?,?,?) ON CONFLICT(tenant,id)
                    DO UPDATE SET kind=excluded.kind,title=excluded.title,text=excluded.text,
                    url=excluded.url,metadata=excluded.metadata""", (
                    tenant, node["id"], node["kind"], node["title"], node["text"], node["url"], json.dumps(metadata)))
                self.db.execute("DELETE FROM chunks WHERE tenant=? AND node_id=?", (tenant, node["id"]))
                self.db.execute("DELETE FROM edges WHERE tenant=? AND src=?", (tenant, node["id"]))
                text = node["text"]
                for start in range(0, len(text), 600):
                    chunk = text[start:start + 600]
                    cid = hashlib.sha256(f"{node['id']}:{start}:{chunk}".encode()).hexdigest()
                    self.db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)", (
                        tenant, cid, node["id"], start, start + len(chunk), chunk,
                        json.dumps(self.encoder.encode(node["title"] + "\n" + chunk)), self.encoder.name))
            for edge in edges:
                if edge["relation"] not in RELATIONS:
                    raise ValueError("invalid relation")
                src, dst = self.node(tenant, edge["src"]), self.node(tenant, edge["dst"])
                src_types, dst_types = EDGE_TYPES[edge["relation"]]
                if not src or not dst or src["kind"] not in src_types or dst["kind"] not in dst_types:
                    raise ValueError("relation endpoints violate typed ontology or tenant boundary")
                self.db.execute("INSERT OR IGNORE INTO edges VALUES (?,?,?,?)", (
                    tenant, edge["src"], edge["relation"], edge["dst"]))

    def node(self, tenant, node_id):
        return self.db.execute("SELECT * FROM nodes WHERE tenant=? AND id=?", (tenant, node_id)).fetchone()

    def resolve_issue(self, tenant, identifier):
        identifier = identifier.upper().replace("-", "")
        rows = self.db.execute("SELECT * FROM nodes WHERE tenant=? AND kind='Issue'", (tenant,)).fetchall()
        return [row for row in rows if str(json.loads(row["metadata"]).get("identifier", "")).upper().replace("-", "") == identifier]

    def authored_by(self, tenant, node_id):
        return self.db.execute("""SELECT n.* FROM edges e JOIN nodes n
            ON n.tenant=e.tenant AND n.id=e.dst
            WHERE e.tenant=? AND e.src=? AND e.relation IN ('AUTHORED_BY','HAS_AUTHOR') LIMIT 20""", (tenant, node_id)).fetchall()

    def scope(self, tenant, node_id):
        rows = self.db.execute("""WITH RECURSIVE scope(id,depth) AS (
            SELECT ?,0 UNION SELECT e.src,s.depth+1 FROM edges e JOIN scope s ON e.dst=s.id
            WHERE e.tenant=? AND e.relation IN ('PART_OF','REPLY_TO') AND s.depth<3)
            SELECT DISTINCT id FROM scope LIMIT 1000""", (node_id, tenant)).fetchall()
        return {row[0] for row in rows}

    def search(self, tenant, query, allowed=None, kind=None, hybrid=False):
        rows = self.db.execute("""SELECT c.*,n.kind,n.title,n.url FROM chunks c
            JOIN nodes n ON n.tenant=c.tenant AND n.id=c.node_id
            WHERE c.tenant=? ORDER BY c.id LIMIT 10001""", (tenant,)).fetchall()
        if len(rows) > 10000:
            raise ValueError("POC scan bound exceeded; use a vector index before scaling")
        if any(row["model"] != self.encoder.name for row in rows):
            raise ValueError("embedding model mismatch; rebuild this database")
        embedding = self.encoder.encode(query)
        lexical = Lexical()
        qlex = lexical.encode(query)
        results = []
        for row in rows:
            if allowed is not None and row["node_id"] not in allowed:
                continue
            if kind and row["kind"] != kind:
                continue
            score = self.encoder.similarity(embedding, json.loads(row["embedding"]))
            if hybrid and self.encoder.is_vector_model:
                score = 0.75 * score + 0.25 * lexical.similarity(qlex, lexical.encode(row["text"]))
            if score > 0:
                results.append({"node_id": row["node_id"], "chunk_id": row["id"], "kind": row["kind"],
                                "text": row["text"], "title": row["title"], "url": row["url"],
                                "start": row["start"], "end": row["end"], "score": round(score, 6)})
        return sorted(results, key=lambda r: (-r["score"], r["chunk_id"]))[:20]

    def delete_source(self, tenant, node_id):
        with self.db:
            self.db.execute("DELETE FROM nodes WHERE tenant=? AND id=?", (tenant, node_id))

    def close(self):
        self.db.close()
