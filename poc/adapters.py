"""Deterministic, tenant-neutral source adapters and opt-in, read-only fetchers.

normalize(source, payload) returns (nodes, edges). Nodes have id, kind, title,
text, url, metadata and optional author (a namespaced person/author node ID).
IDs start with source: and contain entity type to prevent cross-type collisions.
Edges use src/relation/dst; relations are AUTHORED_BY, PART_OF, REPLY_TO,
HAS_TURN, SPOKEN_BY and HAS_AUTHOR. Full bodies are retained, never clipped.
Unknown sources and oversized payloads raise ValueError. No network occurs at
import or normalization. Credentials are read only from the specified env var.

poll_research returns newly enqueued raw paper dictionaries, not graph nodes.
Its research_poll_inbox retains every paper, including previously returned ones,
so a consumer can ingest/retry it independently. Inbox rows and per-topic
published-time checkpoints commit atomically. A one-day overlap plus paper-ID
dedup handles repeat results; this is bounded polling, not an exhaustive archive
or a scheduler. Fetch failures never advance any topic's checkpoint.
"""

import copy
from contextlib import closing
import datetime as dt
import json
import os
import re
import sqlite3
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


MAX_BYTES = 8 * 1024 * 1024
TIMEOUT = 20
MAX_PAGES = 10
UTC = dt.timezone.utc


def _bounded(payload):
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
        raise ValueError("Payload exceeds byte limit")
    return payload


def normalize(source: str, payload: dict) -> tuple[list[dict], list[dict]]:
    """Normalize linear/issues, github/prs, fireflies/transcripts or arxiv/papers.

    Metadata is a deep copy of the raw record. People merge only by explicit
    source ID, never by display name. Fireflies people are always meeting-local;
    arxiv authors without IDs are paper-local by array position. Comment reply
    edges refer only to parents present in that issue/PR's supplied comments.
    """
    keys = {"linear": "issues", "github": "prs", "fireflies": "transcripts",
            "arxiv": "papers"}
    if source not in keys:
        raise ValueError("Unsupported source")
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary")
    _bounded(payload)
    nodes, edges, edge_ids = {}, [], set()

    def ident(kind, value):
        return f"{source}:{kind}:{urllib.parse.quote(str(value), safe='')}"

    def node(nid, kind, title="", text="", url="", raw=None, author=None):
        value = dict(id=nid, kind=kind, title=title or "", text=text or "",
                     url=url or "", metadata=copy.deepcopy(raw or {}))
        if author is not None:
            value["author"] = author
        nodes[nid] = value
        return nid

    def edge(src, relation, dst):
        key = (src, relation, dst)
        if key not in edge_ids:
            edge_ids.add(key)
            edges.append(dict(src=src, relation=relation, dst=dst))

    def person(raw, local=None, kind="Person"):
        if not raw or (raw.get("id") is None and local is None):
            return None
        namespace = "author" if kind == "Author" else "person"
        if local is not None:
            namespace += "-local"
        pid = ident(namespace,
                    local if local is not None else raw["id"])
        return node(pid, kind, raw.get("name") or raw.get("login", ""), raw=raw)

    for record in payload.get(keys[source], []):
        rid = record["id"]
        if source in ("linear", "github"):
            linear = source == "linear"
            root = ident("issue" if linear else "pr", rid)
            author = person(record.get("creator" if linear else "user"))
            node(root, "Issue" if linear else "PullRequest", record.get("title"),
                 record.get("description" if linear else "body"),
                 record.get("url" if linear else "html_url"), record, author)
            if author:
                edge(root, "AUTHORED_BY", author)
            comments = record.get("comments", [])
            comment_ids = {str(c["id"]): ident("comment", c["id"]) for c in comments}
            for comment in comments:
                parent = comment.get("parentId" if linear else "in_reply_to_id")
                cid = comment_ids[str(comment["id"])]
                author = person(comment.get("user"))
                node(cid, "CommentReply" if parent is not None else "Comment",
                     text=comment.get("body"), url=comment.get("url") or comment.get("html_url") or record.get("url" if linear else "html_url"),
                     raw=comment, author=author)
                edge(cid, "PART_OF", root)
                if parent is not None and str(parent) in comment_ids:
                    edge(cid, "REPLY_TO", comment_ids[str(parent)])
                if author:
                    edge(cid, "AUTHORED_BY", author)
        elif source == "fireflies":
            root = ident("meeting", rid)
            sentences = record.get("sentences", [])
            node(root, "Meeting", record.get("title"),
                 "\n".join(s.get("text", "") for s in sentences),
                 record.get("transcript_url"), record)
            speakers = {}
            for speaker in record.get("speakers", []):
                speakers[str(speaker["id"])] = speaker
            for position, sentence in enumerate(sentences):
                sid = sentence.get("speaker_id")
                author = None
                if sid is not None:
                    raw = speakers.get(str(sid), {"id": sid, "name": sentence.get("speaker_name", "")})
                    author = person(raw, json.dumps([str(rid), str(sid)]))
                turn = ident("turn", json.dumps([str(rid), str(sentence.get('index', position))]))
                node(turn, "TranscriptTurn", text=sentence.get("text"), url=record.get("transcript_url"), raw=sentence, author=author)
                edge(root, "HAS_TURN", turn)
                if author:
                    edge(turn, "SPOKEN_BY", author)
        else:
            root = ident("publication", rid)
            node(root, "Publication", record.get("title"), record.get("summary"), record.get("url"), record)
            for index, raw in enumerate(record.get("authors", [])):
                local = None if raw.get("id") is not None else json.dumps([str(rid), index])
                author = person(raw, local, "Author")
                edge(root, "HAS_AUTHOR", author)
    return list(nodes.values()), edges


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect refused")


def _request(url, headers=None, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={
        "User-Agent": "ontology-poc/1.0", **(headers or {})})
    with urllib.request.build_opener(_NoRedirect).open(request, timeout=TIMEOUT) as response:
        content = response.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("Response exceeds byte limit")
    return content


def _token(env):
    token = os.environ.get(env)
    if not token:
        raise ValueError(f"Set credential environment variable {env}")
    return token


def _graphql(url, query, variables, token):
    result = json.loads(_request(url, {"Authorization": token,
                                     "Content-Type": "application/json"},
                                 {"query": query, "variables": variables}))
    if result.get("errors"):
        raise ValueError("Source GraphQL request failed")
    return result["data"]


def fetch_linear(issue_id, token_env="LINEAR_API_KEY"):
    """Fetch one explicit issue and all comments, up to 10 pages of 100.

    Returns the normalize('linear', ...) payload; overflow raises, never silently
    truncates. token_env names the environment variable, not the token itself.
    """
    if not issue_id:
        raise ValueError("Explicit issue ID required")
    query = """query($id:String!,$after:String){issue(id:$id){id identifier title description url
      creator{id name} comments(first:100,after:$after){nodes{id body user{id name} parent{id}}
      pageInfo{hasNextPage endCursor}}}}"""
    after, comments = None, []
    for _ in range(MAX_PAGES):
        issue = _graphql("https://api.linear.app/graphql", query,
                         {"id": str(issue_id), "after": after}, _token(token_env))["issue"]
        if issue is None:
            raise ValueError("Issue not found")
        page = issue.pop("comments")
        for comment in page["nodes"]:
            comment["parentId"] = (comment.pop("parent", None) or {}).get("id")
        comments.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            issue["comments"] = comments
            return _bounded({"issues": [issue]})
        after = page["pageInfo"]["endCursor"]
    raise ValueError("Linear comments exceed page limit")


def fetch_github(repo, number, token_env="GITHUB_TOKEN"):
    """Read one owner/repo PR, issue discussion and review comments via GET only.

    Comment IDs are prefixed issue-/review- because GitHub's two comment tables
    can share numeric IDs. Each collection is limited to 10 pages of 100.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or int(number) < 1:
        raise ValueError("Expected owner/repo and positive PR number")
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {_token(token_env)}"}
    base = f"https://api.github.com/repos/{repo}"
    pr = json.loads(_request(f"{base}/pulls/{int(number)}", headers))
    comments = []
    for kind, path in (("issue", "issues"), ("review", "pulls")):
        for page in range(1, MAX_PAGES + 1):
            batch = json.loads(_request(f"{base}/{path}/{int(number)}/comments?per_page=100&page={page}", headers))
            for comment in batch:
                comment["id"] = f"{kind}-{comment['id']}"
                if comment.get("in_reply_to_id") is not None:
                    comment["in_reply_to_id"] = f"{kind}-{comment['in_reply_to_id']}"
            comments.extend(batch)
            if len(batch) < 100:
                break
        else:
            raise ValueError("GitHub comments exceed page limit")
    pr["comments"] = comments
    return _bounded({"prs": [pr]})


def fetch_fireflies(transcript_id, token_env="FIREFLIES_API_KEY"):
    """Read a single explicit transcript ID; never enumerate meetings."""
    if not transcript_id:
        raise ValueError("Explicit transcript ID required")
    query = """query($id:String!){transcript(id:$id){id title date transcript_url
      sentences{index text speaker_id speaker_name} speakers{id name}}}"""
    record = _graphql("https://api.fireflies.ai/graphql", query,
                      {"id": str(transcript_id)}, f"Bearer {_token(token_env)}")["transcript"]
    if record is None:
        raise ValueError("Transcript not found")
    return _bounded({"transcripts": [record]})


def _date(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return parsed.astimezone(UTC)


def fetch_arxiv(topic, since=None, max_results=20):
    """Return raw papers from a bounded Atom query.

    Cold starts intentionally select only the newest max_results papers, not
    historical backfill. since is an inclusive timezone-aware ISO timestamp;
    incremental queries are oldest-first and overflow raises so a checkpoint
    cannot silently skip records. Narrow the topic on incremental overflow.
    Authors have no stable IDs in Atom, so normalization uses paper-local IDs.
    """
    if not isinstance(topic, str) or not topic.strip() or len(topic) > 500:
        raise ValueError("Topic must contain 1-500 characters")
    if not isinstance(max_results, int) or not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    query = f"({topic})"
    if since:
        query += f" AND submittedDate:[{_date(since).strftime('%Y%m%d%H%M')} TO 999912312359]"
    params = urllib.parse.urlencode(dict(search_query=query, start=0,
        max_results=max_results + 1 if since else max_results,
        sortBy="submittedDate", sortOrder="ascending" if since else "descending"))
    content = _request("https://export.arxiv.org/api/query?" + params)
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("XML declarations refused")
    root = ET.fromstring(content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        get = lambda name: entry.findtext("a:" + name, default="", namespaces=ns)
        pid = get("id")
        if not pid or "/api/errors" in pid:
            raise ValueError("arxiv returned an error entry")
        papers.append(dict(id=pid.rsplit("/abs/", 1)[-1], title=get("title"),
            summary=get("summary"), published=get("published"), url=pid,
            authors=[{"name": a.findtext("a:name", default="", namespaces=ns)}
                     for a in entry.findall("a:author", ns)]))
    if len(papers) > max_results:
        raise ValueError("arxiv result bound exceeded; narrow the topic")
    return papers


def poll_research(db_path, topics, fetcher=fetch_arxiv):
    """Atomically enqueue new papers and checkpoint each exact topic namespace.

    fetcher(topic, since=ISO_or_None, max_results=20) returns raw paper dicts.
    Tables: research_poll_inbox(paper_id PRIMARY KEY, payload JSON), and
    research_poll_checkpoints(topic PRIMARY KEY, published ISO UTC). No inbox
    deletion or graph ingestion happens here. Topics share paper dedup, but not
    checkpoints. Previously enqueued papers can be replayed by reading the inbox.
    """
    topics = list(dict.fromkeys(topics))
    if len(topics) > 20 or any(not isinstance(t, str) or not t.strip() or len(t) > 500 for t in topics):
        raise ValueError("Use at most 20 nonempty topics of at most 500 characters")
    with closing(sqlite3.connect(db_path, timeout=TIMEOUT)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS research_poll_inbox (paper_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS research_poll_checkpoints (topic TEXT PRIMARY KEY, published TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        added = []
        for topic in topics:
            row = db.execute("SELECT published FROM research_poll_checkpoints WHERE topic=?", (topic,)).fetchone()
            checkpoint = _date(row[0]) if row else None
            since = (checkpoint - dt.timedelta(days=1)).isoformat() if checkpoint else None
            papers = fetcher(topic, since=since, max_results=20)
            if not isinstance(papers, list) or len(papers) > 20:
                raise ValueError("Fetcher exceeded paper bound")
            _bounded(papers)
            latest = checkpoint
            for paper in papers:
                if not paper.get("id"):
                    raise ValueError("Paper ID required")
                published = _date(paper["published"])
                latest = max(latest, published) if latest else published
                result = db.execute("INSERT OR IGNORE INTO research_poll_inbox VALUES (?,?)",
                                    (str(paper["id"]), json.dumps(paper, ensure_ascii=False)))
                if result.rowcount:
                    added.append(copy.deepcopy(paper))
            if latest is not None:
                db.execute("INSERT INTO research_poll_checkpoints VALUES (?,?) ON CONFLICT(topic) DO UPDATE SET published=excluded.published",
                           (topic, latest.isoformat()))
        return added
