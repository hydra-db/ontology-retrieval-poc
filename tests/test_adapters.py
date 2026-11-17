"""Offline synthetic fixtures for deterministic adapters and bounded fetchers."""

import copy
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from poc import adapters as a


class NormalizeTests(unittest.TestCase):
    def test_linear_full_text_reply_and_source_identity(self):
        payload = {"issues": [{"id": "i", "identifier": "TEST-1", "title": "Title",
            "description": "first\nsecond\n" + "detail " * 500, "url": "https://example.test/i",
            "creator": {"id": "u", "name": "Person"}, "comments": [
                {"id": "c", "body": "root\ntext", "user": {"id": "u", "name": "Person"}},
                {"id": "r", "body": "reply", "parentId": "c", "user": {"id": "v", "name": "Person"}}]}]}
        before = copy.deepcopy(payload)
        nodes, edges = a.normalize("linear", payload)
        lookup = {n["id"]: n for n in nodes}
        self.assertEqual(lookup["linear:issue:i"]["text"], payload["issues"][0]["description"])
        self.assertEqual(lookup["linear:comment:r"]["kind"], "CommentReply")
        self.assertEqual(lookup["linear:issue:i"]["author"], "linear:person:u")
        self.assertIn(dict(src="linear:comment:r", relation="REPLY_TO", dst="linear:comment:c"), edges)
        self.assertIn(dict(src="linear:comment:c", relation="PART_OF", dst="linear:issue:i"), edges)
        self.assertIn(dict(src="linear:issue:i", relation="AUTHORED_BY", dst="linear:person:u"), edges)
        self.assertEqual(len([n for n in nodes if n["kind"] == "Person"]), 2)
        self.assertEqual(before, payload)
        lookup["linear:issue:i"]["metadata"]["creator"]["name"] = "changed"
        self.assertEqual(before, payload)
        self.assertEqual(a.normalize("linear", payload), a.normalize("linear", payload))
        for n in nodes:
            self.assertTrue({"id", "kind", "title", "text", "url", "metadata"} <= n.keys())
        self.assertTrue(all(e["src"] in lookup and e["dst"] in lookup for e in edges))

    def test_github_missing_user_and_missing_parent(self):
        nodes, edges = a.normalize("github", {"prs": [{"id": 1, "number": 7,
            "title": "PR", "body": "full body", "html_url": "https://example.test/pr",
            "user": {"id": 1, "login": "dev"}, "comments": [
                {"id": 1, "body": "root", "user": None},
                {"id": 2, "body": "reply", "in_reply_to_id": 1},
                {"id": 3, "body": "orphan", "in_reply_to_id": 99}]}]})
        self.assertEqual(len({n["id"] for n in nodes}), len(nodes))
        pr = next(n for n in nodes if n["kind"] == "PullRequest")
        self.assertEqual(pr["text"], "full body")
        self.assertEqual(pr["metadata"]["number"], 7)
        self.assertEqual(len([e for e in edges if e["relation"] == "REPLY_TO"]), 1)
        self.assertNotIn("author", next(n for n in nodes if n["kind"] == "Comment"))

    def test_fireflies_speaker_identity_is_meeting_local(self):
        records = [{"id": mid, "title": "Meeting", "date": 1, "transcript_url": "url",
            "speakers": [{"id": 0, "name": "Same Name"}],
            "sentences": [{"index": 0, "text": "verbatim\ntext", "speaker_id": 0},
                          {"index": 1, "text": "unknown"}]} for mid in ("m1", "m2")]
        nodes, edges = a.normalize("fireflies", {"transcripts": records})
        people = [n for n in nodes if n["kind"] == "Person"]
        self.assertEqual(len(people), 2)
        self.assertNotEqual(people[0]["id"], people[1]["id"])
        self.assertEqual(len([n for n in nodes if n["kind"] == "TranscriptTurn"]), 4)
        self.assertEqual(next(n for n in nodes if n["kind"] == "Meeting")["text"], "verbatim\ntext\nunknown")
        self.assertEqual(len([e for e in edges if e["relation"] == "SPOKEN_BY"]), 2)

    def test_arxiv_does_not_merge_names_but_merges_explicit_ids(self):
        papers = [{"id": pid, "title": "Paper", "summary": "full\nsummary",
            "published": "2026-01-01T00:00:00Z", "url": "url", "authors": [
                {"name": "Name"}, {"name": "Name"}, {"id": "stable", "name": "Named"}]} for pid in ("p1", "p2")]
        nodes, edges = a.normalize("arxiv", {"papers": papers})
        self.assertEqual(len([n for n in nodes if n["kind"] == "Author"]), 5)
        self.assertEqual(len([e for e in edges if e["relation"] == "HAS_AUTHOR"]), 6)
        self.assertEqual(next(n for n in nodes if n["kind"] == "Publication")["text"], "full\nsummary")

    def test_unknown_and_oversized_payload_rejected(self):
        with self.assertRaises(ValueError):
            a.normalize("unknown", {})
        with patch.object(a, "MAX_BYTES", 10), self.assertRaises(ValueError):
            a.normalize("linear", {"issues": ["too much data"]})


class FetchTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"TEST_TOKEN": "synthetic-test-token"})
        env.start()
        self.addCleanup(env.stop)

    def test_linear_paginates_and_maps_parent(self):
        def issue(cursor, more, cid, parent=None):
            return {"issue": {"id": "i", "title": "Issue", "comments": {
                "nodes": [{"id": cid, "body": "body", "parent": parent}],
                "pageInfo": {"hasNextPage": more, "endCursor": cursor}}}}
        with patch.object(a, "_graphql", side_effect=[issue("next", True, "c"), issue(None, False, "r", {"id": "c"})]) as call:
            payload = a.fetch_linear("explicit", "TEST_TOKEN")
        self.assertEqual(call.call_args_list[1].args[2], {"id": "explicit", "after": "next"})
        self.assertEqual(payload["issues"][0]["comments"][1]["parentId"], "c")
        self.assertEqual(call.call_args.args[0], "https://api.linear.app/graphql")

    def test_github_includes_both_comment_streams_without_id_collision(self):
        replies = [{"id": 1, "title": "PR"}, [{"id": 8, "body": "discussion"}],
                   [{"id": 8, "body": "review"}, {"id": 9, "in_reply_to_id": 8}]]
        with patch.object(a, "_request", side_effect=[json.dumps(r).encode() for r in replies]) as call:
            payload = a.fetch_github("owner/repo", 1, "TEST_TOKEN")
        comments = payload["prs"][0]["comments"]
        self.assertEqual([c["id"] for c in comments], ["issue-8", "review-8", "review-9"])
        self.assertEqual(comments[-1]["in_reply_to_id"], "review-8")
        self.assertTrue(all(c.args[0].startswith("https://api.github.com/repos/owner/repo/") for c in call.call_args_list))
        self.assertTrue(all(len(c.args) == 2 for c in call.call_args_list))
        with self.assertRaises(ValueError):
            a.fetch_github("https://evil.test", 1, "TEST_TOKEN")

    def test_fireflies_explicit_id(self):
        with patch.object(a, "_graphql", return_value={"transcript": {"id": "meeting"}}) as call:
            self.assertEqual(a.fetch_fireflies("meeting", "TEST_TOKEN"), {"transcripts": [{"id": "meeting"}]})
        self.assertEqual(call.call_args.args[0], "https://api.fireflies.ai/graphql")
        self.assertEqual(call.call_args.args[2], {"id": "meeting"})
        with self.assertRaises(ValueError):
            a.fetch_fireflies("", "TEST_TOKEN")

    def test_credentials_missing_and_graphql_error(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            a.fetch_linear("id")
        with patch.object(a, "_request", return_value=b'{"errors":[{"message":"private"}]}'):
            with self.assertRaisesRegex(ValueError, "Source GraphQL request failed"):
                a.fetch_fireflies("id", "TEST_TOKEN")

    def test_page_overflow_is_not_truncated(self):
        page = {"issue": {"id": "i", "comments": {"nodes": [], "pageInfo": {
            "hasNextPage": True, "endCursor": "x"}}}}
        with patch.object(a, "MAX_PAGES", 1), patch.object(a, "_graphql", return_value=page):
            with self.assertRaisesRegex(ValueError, "page limit"):
                a.fetch_linear("id", "TEST_TOKEN")

    def test_arxiv_atom_preserves_text_and_bounds_query(self):
        atom = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>https://arxiv.org/abs/1234.5678v1</id><title> A title </title>
          <summary>first\nsecond</summary><published>2026-01-02T00:00:00Z</published>
          <author><name>Same Name</name></author></entry></feed>'''
        with patch.object(a, "_request", return_value=atom) as call:
            papers = a.fetch_arxiv("cat:cs.AI", since="2026-01-01T00:00:00Z")
        self.assertEqual(papers[0]["summary"], "first\nsecond")
        self.assertEqual(papers[0]["title"], " A title ")
        self.assertEqual(papers[0]["authors"], [{"name": "Same Name"}])
        url = call.call_args.args[0]
        self.assertTrue(url.startswith("https://export.arxiv.org/api/query?"))
        self.assertIn("max_results=21", url)
        self.assertIn("202601010000", url)
        with self.assertRaises(ValueError):
            a.fetch_arxiv("topic", max_results=101)
        entry = atom[atom.index(b"<entry>"):atom.index(b"</entry>") + len(b"</entry>")]
        overflow = atom.replace(b"</feed>", entry + b"</feed>")
        with patch.object(a, "_request", return_value=overflow):
            with self.assertRaises(ValueError):
                a.fetch_arxiv("topic", since="2026-01-01T00:00:00Z", max_results=1)
        with patch.object(a, "_request", return_value=atom) as call:
            a.fetch_arxiv("topic")
        self.assertIn("max_results=20", call.call_args.args[0])
        self.assertIn("sortOrder=descending", call.call_args.args[0])

    def test_http_limits_timeout_and_redirects(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self, limit):
                self.limit = limit
                return b"x" * limit
        response = Response()
        with patch.object(a.urllib.request, "build_opener") as opener, patch.object(a, "MAX_BYTES", 4):
            opener.return_value.open.return_value = response
            with self.assertRaises(ValueError):
                a._request("https://api.github.com/test")
            self.assertEqual(response.limit, 5)
            self.assertEqual(opener.return_value.open.call_args.kwargs["timeout"], 20)
        with self.assertRaises(ValueError):
            a._NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test")

    def test_arxiv_rejects_xml_entities_and_error_entries(self):
        for content in (b'<!DOCTYPE feed [<!ENTITY x "bad">]><feed/>',
                        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/api/errors#bad</id></entry></feed>'):
            with patch.object(a, "_request", return_value=content), self.assertRaises(ValueError):
                a.fetch_arxiv("topic")


class PollTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = os.path.join(directory.name, "poll.sqlite")
        self.paper = dict(id="p1", title="Title", summary="full text", published="2026-01-02T00:00:00Z", authors=[])

    def rows(self, table):
        with sqlite3.connect(self.db) as db:
            return db.execute("SELECT * FROM " + table).fetchall()

    def test_durable_inbox_dedup_topic_checkpoint_and_overlap(self):
        calls = []
        def fetch(topic, **kwargs):
            calls.append((topic, kwargs))
            return [self.paper]
        self.assertEqual(a.poll_research(self.db, ["a", "b"], fetch), [self.paper])
        self.assertEqual(a.poll_research(self.db, ["a"], fetch), [])
        self.assertEqual(len(self.rows("research_poll_checkpoints")), 2)
        self.assertEqual(json.loads(self.rows("research_poll_inbox")[0][1]), self.paper)
        self.assertEqual(calls[0][1], {"since": None, "max_results": 20})
        self.assertEqual(calls[-1][1]["since"], "2026-01-01T00:00:00+00:00")

    def test_failure_rolls_back_entire_poll(self):
        def fetch(topic, **kwargs):
            if topic == "fail":
                raise RuntimeError("source offline")
            return [self.paper]
        with self.assertRaises(RuntimeError):
            a.poll_research(self.db, ["good", "fail"], fetch)
        self.assertEqual(self.rows("research_poll_inbox"), [])
        self.assertEqual(self.rows("research_poll_checkpoints"), [])

    def test_invalid_paper_rolls_back_and_empty_does_not_checkpoint(self):
        a.poll_research(self.db, ["empty"], lambda *args, **kwargs: [])
        self.assertEqual(self.rows("research_poll_checkpoints"), [])
        bad = dict(self.paper, published="not a timestamp")
        with self.assertRaises(ValueError):
            a.poll_research(self.db, ["bad"], lambda *args, **kwargs: [self.paper, bad])
        self.assertEqual(self.rows("research_poll_inbox"), [])

    def test_checkpoint_never_regresses(self):
        a.poll_research(self.db, ["a"], lambda *args, **kwargs: [self.paper])
        older = dict(self.paper, id="older", published="2026-01-01T12:00:00Z")
        self.assertEqual(a.poll_research(self.db, ["a"], lambda *args, **kwargs: [older]), [older])
        self.assertEqual(self.rows("research_poll_checkpoints")[0][1], "2026-01-02T00:00:00+00:00")

    def test_failure_preserves_existing_inbox_and_checkpoint(self):
        a.poll_research(self.db, ["a"], lambda *args, **kwargs: [self.paper])
        before = self.rows("research_poll_checkpoints")
        newer = dict(self.paper, id="new", published="2026-01-03T00:00:00Z")
        def fetch(topic, **kwargs):
            if topic == "fail":
                raise RuntimeError("offline")
            return [newer]
        with self.assertRaises(RuntimeError):
            a.poll_research(self.db, ["a", "fail"], fetch)
        self.assertEqual(self.rows("research_poll_checkpoints"), before)
        self.assertEqual(len(self.rows("research_poll_inbox")), 1)

    def test_fetcher_bound_and_topic_bound(self):
        with self.assertRaises(ValueError):
            a.poll_research(self.db, ["a"], lambda *args, **kwargs: [self.paper] * 21)
        self.assertEqual(self.rows("research_poll_inbox"), [])
        with self.assertRaises(ValueError):
            a.poll_research(self.db, [str(i) for i in range(21)])


if __name__ == "__main__":
    unittest.main()
