import json
import os
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

from poc.adapters import normalize
from poc.preferences import Preferences
from poc.retrieval import Engine
from poc.store import CrossEncoderReranker, SentenceEncoder, Store


@unittest.skipUnless(os.environ.get("POC_TEST_MODELS") == "1", "opt-in downloaded local models")
class LocalModelTest(unittest.TestCase):
    def test_real_vector_and_cross_encoder_work_offline(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
            encoder = SentenceEncoder(root / "models/encoder")
            reranker = CrossEncoderReranker(str(root / "models/reranker"))
            self.assertEqual(len(encoder.encode("semantic retrieval")), 384)
            store = Store(encoder=encoder)
            for source, payload in json.loads((root / "fixtures/sources.json").read_text()).items():
                store.ingest("demo", *normalize(source, payload))
            prefs = Preferences(store)
            for turn in json.loads((root / "fixtures/turns.json").read_text()):
                prefs.add_turn("demo", "alice", turn)
            prefs.learn("demo", "alice")
            engine = Engine(store, prefs, reranker)
            result = engine.query("demo", "alice", "What is turbolay?")
            self.assertEqual(result["route"], "personalized")
            self.assertEqual(result["stores"], ["preferences", "vector"])
            self.assertTrue(result["results"])
            self.assertIn("reranker_score", result["results"][0])
            self.assertTrue(all(store.node("demo", r["node_id"]) for r in result["results"]))
            store.close()
