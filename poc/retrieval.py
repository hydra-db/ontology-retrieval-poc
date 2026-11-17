import re
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Decision:
    route: str
    confidence: float
    reason: str
    identifier: str | None = None
    kind: str | None = None


class Router:
    """Closed grammar, no generated plans. Precision/abstention beats coverage."""

    def decide(self, query):
        q = query.strip().lower()
        if not q or len(q) > 2000:
            return Decision("unsupported", 1, "empty or oversized query")
        ids = re.findall(r"\b([a-z]{2,10})-?(\d{1,9})\b", q)
        identifier = f"{ids[0][0].upper()}-{ids[0][1]}" if len(ids) == 1 else None
        if len(ids) > 1:
            return Decision("ambiguous", 1, "one explicit issue ID per query")
        if re.fullmatch(r"(?:who (?:authored|created|wrote)|author of|creator of) (?:the )?(?:linear )?(?:ticket |issue )?[a-z]{2,10}-?\d{1,9}\??", q):
            return Decision("ontology", 1, "fixed authored-by template", identifier)
        if re.search(r"\b(delete|ignore|drop|update|execute|password|secret|weather)\b", q):
            return Decision("unsupported", 1, "outside read-only search grammar")
        if q.startswith(("who ", "how many ", "why ", "when ")):
            return Decision("ambiguous", 1, "unsupported relation or unresolved person; ask for explicit ID")
        kind = "Publication" if q.startswith(("papers about ", "research about ")) else None
        if not q.startswith(("search ", "find ", "papers about ", "research about ", "what is ")):
            return Decision("unsupported", 1, "no supported intent; use search/find or an explicit authorship question")
        if identifier:
            return Decision("graph_search", 0.98, "explicit issue scope; fixed bounded descendants", identifier)
        return Decision("semantic", 0.9, "bounded content search", kind=kind)


class Engine:
    def __init__(self, store, preferences, reranker=None):
        self.store, self.preferences, self.reranker = store, preferences, reranker
        self.router = Router()

    def query(self, tenant, user, query, personalized=True, limit=5):
        start = time.perf_counter_ns()
        if not tenant or not user or not 1 <= limit <= 20:
            raise ValueError("explicit tenant/user and limit 1..20 required")
        decision = self.router.decide(query)
        trace = [asdict(decision)]
        stores, results, prefs = [], [], []
        status = "ok"
        route = decision.route
        effective_query = query
        if route in {"unsupported", "ambiguous"}:
            status = route
        elif route in {"ontology", "graph_search"}:
            stores.append("ontology")
            entities = self.store.resolve_issue(tenant, decision.identifier)
            if len(entities) != 1:
                status = "not_found" if not entities else "ambiguous"
                trace.append({"entity_count": len(entities), "fallback": "none; never broaden unresolved graph scope"})
            elif route == "ontology":
                results = [{"node_id": row["id"], "kind": row["kind"], "text": row["title"],
                            "url": row["url"], "evidence_node_id": entities[0]["id"], "evidence_url": entities[0]["url"],
                            "relation": "AUTHORED_BY"} for row in self.store.authored_by(tenant, entities[0]["id"])]
                status = "ok" if results else "not_found"
            else:
                allowed = self.store.scope(tenant, entities[0]["id"])
                stores.append("vector" if self.store.encoder.is_vector_model else "lexical")
                results = self.store.search(tenant, query, allowed=allowed, hybrid=True)
                trace.append({"template": "issue_descendants_depth_3", "scope_nodes": len(allowed), "hybrid_weights": [0.75, 0.25] if self.store.encoder.is_vector_model else None})
        else:
            expanded = query
            if personalized:
                stores.append("preferences")
                prefs = self.preferences.relevant(tenant, user, query)
                for pref in prefs:
                    expanded = re.sub(r"\b" + re.escape(pref["key"]) + r"\b", pref["value"], expanded, flags=re.I)
                trace.append({"preferences": [{k: p[k] for k in ("id", "key", "value", "quote", "turn_id", "confidence")} for p in prefs],
                              "fallback": "unpersonalized" if not prefs else None})
            if prefs:
                route = "personalized"
            effective_query = expanded
            stores.append("vector" if self.store.encoder.is_vector_model else "lexical")
            results = self.store.search(tenant, expanded, kind=decision.kind)
            trace.append({"expanded_query": expanded, "candidate_limit": 20, "kind_filter": decision.kind})
        if self.reranker and results and route != "ontology":
            scores = self.reranker.score(effective_query, [r["text"] for r in results])
            for result, score in zip(results, scores, strict=True):
                result["reranker_score"] = float(score)
            results.sort(key=lambda r: -r["reranker_score"])
            trace.append({"reranked": len(results), "model": "local_cross_encoder"})
        if status == "ok" and not results:
            status = "not_found"
        return {"status": status, "route": route, "stores": stores, "results": results[:limit], "trace": trace,
                "latency_ms": round((time.perf_counter_ns() - start) / 1e6, 4),
                "search_backend": self.store.encoder.name, "network_calls": 0, "generative_calls": 0}
