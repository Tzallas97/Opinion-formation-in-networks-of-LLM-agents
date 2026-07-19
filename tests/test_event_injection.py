#!/usr/bin/env python3
"""Unit tests for event_injection. No LLM, no simulator.

Run:  python tests/test_event_injection.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import event_injection as ev  # noqa: E402


class FakeAgent:
    def __init__(self, name, belief=0):
        self.agent_name = name
        self.current_belief = belief
        self.stored = []  # (text, pinned)


def test_broadcast_hits_every_agent_once():
    agents = [FakeAgent(f"A{i}", belief=i - 2) for i in range(5)]
    calls = []

    def react(agent, prompt):
        calls.append(agent.agent_name)
        assert "EVENT:" in prompt and agent.agent_name in prompt
        return f"{agent.agent_name} reacts"

    def store(agent, text, pinned):
        agent.stored.append((text, pinned))

    rows = ev.broadcast_event(agents, "Moon landing", 300, react, store, persist="reaction")
    assert calls == [a.agent_name for a in agents], "not every agent asked exactly once"
    assert len(rows) == 5
    assert all(r["step"] == 300 for r in rows)
    assert all(a.stored and a.stored[0][1] is False for a in agents), "reaction should not pin"
    print("OK: broadcast asks every agent once and stores an un-pinned reaction")


def test_persist_conditions():
    def react(agent, prompt):
        return "my take"

    # reaction: stores the reaction, not pinned
    a = FakeAgent("X")
    ev.broadcast_event([a], "E", 10, react, lambda ag, t, p: ag.stored.append((t, p)), persist="reaction")
    assert a.stored[0] == ("my take", False)

    # headline: stores the event text, pinned
    b = FakeAgent("X")
    ev.broadcast_event([b], "E", 10, react, lambda ag, t, p: ag.stored.append((t, p)), persist="headline")
    assert b.stored[0] == ("E", True), b.stored

    # both: stores event + reaction, pinned
    c = FakeAgent("X")
    ev.broadcast_event([c], "E", 10, react, lambda ag, t, p: ag.stored.append((t, p)), persist="both")
    assert c.stored[0][1] is True and "E" in c.stored[0][0] and "my take" in c.stored[0][0]
    print("OK: reaction/headline/both persist conditions store and pin correctly")


def test_should_pin():
    assert ev.should_pin("headline") and ev.should_pin("both")
    assert not ev.should_pin("reaction") and not ev.should_pin("")
    print("OK: should_pin matches the three conditions")


def test_relevance_cosine():
    # a toy embedder: identical text -> identical vector -> cosine 1
    table = {"same": [1.0, 0.0, 0.0], "near": [0.9, 0.1, 0.0], "far": [0.0, 0.0, 1.0]}
    embed = lambda texts: [table[t] for t in texts]
    assert abs(ev.event_relevance("same", "same", embed) - 1.0) < 1e-9
    assert ev.event_relevance("far", "same", embed) == 0.0
    assert ev.event_relevance("near", "same", embed) > 0.9
    assert ev.event_relevance("x", "y", None) is None, "no embedder -> None, run still works"
    print("OK: relevance is cosine in embedding space, None without an embedder")


def test_missing_template_falls_back():
    tmpl = ev.load_event_template("/no/such/dir")
    assert tmpl == ev.DEFAULT_EVENT_TEMPLATE
    prompt = ev.build_event_prompt(tmpl, "Ada", "Rains on Mars", 1)
    assert "Ada" in prompt and "Rains on Mars" in prompt
    print("OK: missing template falls back to the built-in default")


def test_one_agent_failure_does_not_abort():
    agents = [FakeAgent("good1"), FakeAgent("bad"), FakeAgent("good2")]

    def react(agent, prompt):
        if agent.agent_name == "bad":
            raise RuntimeError("model timeout")
        return "fine"

    rows = ev.broadcast_event(agents, "E", 5, react, lambda a, t, p: None)
    assert len(rows) == 3, "a single failure aborted the broadcast"
    assert rows[1]["reaction"] == "" and rows[0]["reaction"] == "fine"
    print("OK: one agent's model failure does not abort the shock")


def main():
    test_broadcast_hits_every_agent_once()
    test_persist_conditions()
    test_should_pin()
    test_relevance_cosine()
    test_missing_template_falls_back()
    test_one_agent_failure_does_not_abort()


if __name__ == "__main__":
    main()
