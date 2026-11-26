"""Offline integrity regressions: inert records, fake transports, temporary copies."""

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from benchmarks import collect, registry, validate
from benchmarks import contracts as c
from benchmarks import scoring_v2 as s

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
CORPUS = "R1-FALSE-REFUSAL-443-v1"


def response(cid=CORPUS, position=0):
    rid = list(s.row_map(registry.rows(BENCH, cid)))[position]
    return {
        "schema": "blackfrost.benchmark-response.v2",
        "corpus_id": cid,
        "corpus_sha256": registry.load(BENCH)[cid]["sha256"],
        "row_id": rid,
        "run_id": "inert-test",
        "model_id": "synthetic",
        "model_digest": "a" * 64,
        "response_text": "SYNTHETIC",
        "answer_text": "inert answer",
        "reasoning_text": "",
        "finish_reason": "stop",
        "status": "ok",
        "requested_settings": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 10,
            "seed": None,
            "chat_template": "unknown",
            "thinking_mode": "on",
        },
        "observed_settings": {
            "model_id": "synthetic",
            "chat_template": None,
            "thinking_mode": "unknown",
        },
        "collector_version": "synthetic",
        "sequence": position,
    }


def judgment(r, value="engaged"):
    return {
        "schema": "blackfrost.benchmark-judgment.v2",
        **{k: r[k] for k in s.IDENTITY},
        "row_id": r["row_id"],
        "response_sha256": c.digest(r),
        "rubric_sha256": c.sha256(BENCH / "protocol/SCORING-v2.md"),
        "judgment": value,
        "evaluator": {"kind": "human", "id": "inert-reviewer", "version": "1"},
        "notes": "synthetic",
    }


def files(tmp_path, responses, judgments):
    rp, jp = tmp_path / "responses.jsonl", tmp_path / "judgments.jsonl"
    for path, records in [(rp, responses), (jp, judgments)]:
        path.write_bytes(b"".join(c.canonical(r) + b"\n" for r in records))
    return rp, jp


def test_population_metrics_and_unfinished(tmp_path):
    rows = registry.rows(BENCH, CORPUS)
    positions = [
        0,
        1,
        next(i for i, r in enumerate(rows) if r["label"] == "safe"),
        next(i for i, r in enumerate(rows) if r["label"] == "unsafe"),
        2,
        3,
        4,
    ]
    rs = [response(position=i) for i in positions]
    rs[4].update(status="truncated", finish_reason="length")
    js = [
        judgment(r, v)
        for r, v in zip(
            rs,
            ["engaged", "refused", "refused", "refused", "engaged", "degenerate", "needs_review"],
            strict=True,
        )
    ]
    out = s.score(CORPUS, *files(tmp_path, rs, js))
    assert out["states"] == {
        "ENGAGED": 1,
        "REFUSED": 3,
        "ERROR": 1,
        "DEGENERATE": 1,
        "NEEDS_REVIEW": 1,
        "MISSING": 436,
    }
    assert out["rates"]["harmful_engagement"] == {
        "numerator": 1,
        "denominator": 2,
        "total": 295,
        "unresolved": 293,
        "value": 0.5,
    }
    assert out["rates"]["benign_false_refusal"] == {
        "numerator": 1,
        "denominator": 1,
        "total": 75,
        "unresolved": 74,
        "value": 1.0,
    }
    assert out["coverage"]["collection_complete"] is False
    assert out["release_eligible"] is False


@pytest.mark.parametrize("cid", sorted(registry.IDS))
def test_all_corpora_cli_and_categories(tmp_path, cid):
    r = response(cid)
    out = s.score(cid, *files(tmp_path, [r], [judgment(r)]))
    assert out["states"]["ENGAGED"] == 1
    assert not ({"PASS", "FAIL"} & out["states"].keys())
    if cid != CORPUS:
        assert out["rates"] == {}
    proc = subprocess.run(
        [sys.executable, str(BENCH / "score.py"), "--corpus", cid, "--not-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    c.validate(result, "metrics")
    assert result["states"] == {"NOT_RUN": registry.IDS[cid]}


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer_text", "tampered"),
        ("reasoning_text", "tampered"),
        ("response_text", "tampered"),
        ("run_id", "other"),
        ("model_digest", "b" * 64),
        ("model_id", "other"),
    ],
)
def test_stale_judgment_rejected(tmp_path, field, value):
    r = response()
    j = judgment(r)
    r[field] = value
    with pytest.raises(ValueError, match="binding"):
        s.score(CORPUS, *files(tmp_path, [r], [j]))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(extra=True),
        lambda r: r.pop("answer_text"),
        lambda r: r.update(answer_text=[]),
        lambda r: r.update(status="bogus"),
        lambda r: r.update(sequence=True),
        lambda r: r.update(finish_reason="length"),
        lambda r: r["requested_settings"].update(top_p=3),
        lambda r: r["observed_settings"].update(thinking_mode="on"),
    ],
)
def test_invalid_responses(tmp_path, mutate):
    r = response()
    mutate(r)
    with pytest.raises(ValueError):
        s.score(CORPUS, *files(tmp_path, [r], []))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda j: j["evaluator"].update(kind="deterministic"),
        lambda j: j["evaluator"].update(kind="llm"),
        lambda j: j.update(rubric_sha256="0" * 64),
        lambda j: j.update(response_sha256="0" * 64),
        lambda j: j.update(judgment="pass"),
        lambda j: j.pop("run_id"),
    ],
)
def test_invalid_judgments(tmp_path, mutate):
    r = response()
    j = judgment(r)
    mutate(j)
    with pytest.raises(ValueError):
        s.score(CORPUS, *files(tmp_path, [r], [j]))


def test_mixed_duplicate_orphan_and_swapped(tmp_path):
    a, b = response(), response(position=1)
    for rs, js in [
        ([a, a], []),
        ([a], [judgment(a), judgment(a)]),
        ([], [judgment(a)]),
        ([a, b], [{**judgment(a), "row_id": b["row_id"]}]),
        ([a, {**b, "run_id": "other"}], []),
        ([a, {**b, "sequence": 0}], []),
    ]:
        with pytest.raises(ValueError):
            s.score(CORPUS, *files(tmp_path, rs, js))


@pytest.mark.parametrize(
    "raw",
    [
        '{"x":1,"x":2}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        '{"x":"\\ud800"}',
        "[]",
        "null",
        "",
    ],
)
def test_strict_jsonl(tmp_path, raw):
    p = tmp_path / "bad.jsonl"
    p.write_text(raw + "\n")
    with pytest.raises(ValueError):
        c.jsonl(p, "responses")


def test_local_schema_no_resolution(tmp_path, monkeypatch):
    root = tmp_path / "bench"
    (root / "protocol").mkdir(parents=True)
    (root / "protocol/responses.v2.schema.json").write_text(
        '{"$ref":"https://invalid.example/schema"}'
    )
    monkeypatch.setattr(c, "ROOT", root)
    with pytest.raises(ValueError, match="references forbidden"):
        c.validate({}, "responses")


def args(tmp_path, cid="CYBER-REASONING-10-v1"):
    return argparse.Namespace(
        corpus=cid,
        endpoint="https://invalid.example/v1/chat/completions",
        model="synthetic",
        model_digest="a" * 64,
        run_id="inert-test",
        out=tmp_path / "output.jsonl",
        temperature=1.0,
        top_p=0.95,
        max_tokens=10,
        seed=None,
        chat_template="intent",
        thinking_mode="on",
        header=[],
    )


def good_raw():
    return {
        "model": "synthetic",
        "choices": [{"message": {"content": "inert"}, "finish_reason": "stop"}],
    }


def test_collection_completion_and_tamper(tmp_path, monkeypatch):
    a = args(tmp_path)
    payloads = []

    def transport(endpoint, payload, headers):
        payloads.append(payload)
        return good_raw()

    monkeypatch.setattr(collect, "chat_completion", transport)
    receipt = collect.collect(a)
    cp = Path(str(a.out) + ".complete.json")
    assert receipt["responses_sha256"] == c.sha256(a.out) and len(receipt["responses_sha256"]) == 64
    assert receipt["errors"] == 0
    records = c.jsonl(a.out, "responses")
    assert len(records) == 10 and len(payloads) == 10
    assert all("thinking_mode" not in p and "chat_template" not in p for p in payloads)
    assert records[0]["requested_settings"]["thinking_mode"] == "on"
    assert records[0]["observed_settings"]["thinking_mode"] == "unknown"
    _, jp = files(tmp_path, [], [judgment(r) for r in records])
    out = s.score(a.corpus, a.out, jp, completion_path=cp)
    assert out["coverage"]["collection_complete"]
    with pytest.raises(ValueError):
        collect.collect(a)
    a.out.write_bytes(a.out.read_bytes() + b" ")
    with pytest.raises(ValueError):
        s.score(a.corpus, a.out, jp, completion_path=cp)


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_collection_scorer_round_trip_preserves_unicode_line_separator(
    tmp_path, monkeypatch, separator
):
    a = args(tmp_path)
    text = f"before{separator}after"

    monkeypatch.setattr(
        collect,
        "chat_completion",
        lambda *unused: {
            "model": "synthetic",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        },
    )
    receipt = collect.collect(a)
    responses = c.jsonl(a.out, "responses")
    assert receipt["errors"] == 0
    assert len(responses) == 10
    assert all(record["answer_text"] == text for record in responses)

    _, judgments = files(tmp_path, [], [judgment(record) for record in responses])
    scored = s.score(
        a.corpus, a.out, judgments, completion_path=Path(str(a.out) + ".complete.json")
    )
    assert scored["coverage"]["collection_complete"]
    assert scored["states"]["ENGAGED"] == 10


def test_interruption_persists_incrementally(tmp_path, monkeypatch):
    a = args(tmp_path)
    calls = 0

    def transport(*unused):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert len(c.jsonl(a.out, "responses")) == 1
            raise KeyboardInterrupt
        return good_raw()

    monkeypatch.setattr(collect, "chat_completion", transport)
    with pytest.raises(KeyboardInterrupt):
        collect.collect(a)
    assert len(c.jsonl(a.out, "responses")) == 1
    assert not Path(str(a.out) + ".complete.json").exists()
    with pytest.raises(FileExistsError):
        collect.collect(a)
    assert calls == 2


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"choices": "bad"},
        {"choices": [None]},
        {"choices": [{"message": []}]},
        {"choices": [{"message": {"content": []}}]},
        {"choices": [{"message": {"content": "x", "reasoning": {}}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "x"}, "finish_reason": []}]},
        {**good_raw(), "model": "other"},
    ],
)
def test_malformed_transport_results_record_errors(tmp_path, monkeypatch, raw):
    a = args(tmp_path)
    monkeypatch.setattr(collect, "chat_completion", lambda *unused: raw)
    result = collect.collect(a)
    assert result["errors"] == 10
    assert all(r["status"] == "error" for r in c.jsonl(a.out, "responses"))


@pytest.mark.parametrize(
    "url",
    [
        "file:///secret",
        "https://user:secret@example.com/x",
        "https://example.com/x?token=secret",
        "https://example.com/#x",
        "https://example.com:bad/x",
        "https://exa\nmple.com/x",
        "//example.com",
    ],
)
def test_endpoint_rejection(url):
    with pytest.raises(ValueError, match="invalid endpoint"):
        collect.endpoint_check(url)


@pytest.mark.parametrize(
    "headers",
    [
        ["bad secret"],
        ["Authorization: secret", "authorization: other"],
        ["X-Test: a\nb"],
        ["Host: other"],
        ["Content-Length: 1"],
    ],
)
def test_header_rejection(headers):
    with pytest.raises(ValueError):
        collect.parse_headers(headers)


def test_bounded_read_and_redirect_handler(monkeypatch):
    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def read(self, n):
            assert n == collect.MAX_RESPONSE_BYTES + 1
            return b"x" * n

    class Opener:
        def open(self, *unused, **kwargs):
            return Reply()

    handlers = []

    def build(*values):
        handlers.extend(values)
        return Opener()

    monkeypatch.setattr(collect.urllib.request, "build_opener", build)
    with pytest.raises(ValueError, match="byte limit"):
        collect.chat_completion("https://invalid.example/x", {}, {})
    redirect = next(h for h in handlers if isinstance(h, collect.NoRedirect))
    for code in (301, 302, 303, 307, 308):
        with pytest.raises(ValueError, match="redirect"):
            redirect.redirect_request(None, None, code, None, None, "https://other.invalid")
    assert (
        next(h for h in handlers if isinstance(h, collect.urllib.request.ProxyHandler)).proxies
        == {}
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["corpora"].append(copy.deepcopy(r["corpora"][0])),
        lambda r: r["corpora"][0].update(id="unknown"),
        lambda r: r["corpora"][0].update(file="../secret"),
        lambda r: r["corpora"][0].update(file="/secret"),
        lambda r: r["corpora"][0].update(rows=True),
        lambda r: r["corpora"][0].update(sha256="bad"),
        lambda r: r["corpora"][0].update(extra=True),
    ],
)
def test_registry_failures_are_controlled(tmp_path, monkeypatch, change):
    root = tmp_path / "bench"
    shutil.copytree(BENCH, root)
    p = root / "REGISTRY.yaml"
    raw = yaml.safe_load(p.read_text())
    change(raw)
    p.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(validate, "ROOT", root)
    assert validate.check()


def test_duplicate_yaml_and_symlink_escape(tmp_path):
    root = tmp_path / "bench"
    shutil.copytree(BENCH, root)
    p = root / "REGISTRY.yaml"
    original = p.read_text()
    p.write_text(original + "\nschema: duplicate\n")
    with pytest.raises(ValueError):
        registry.load(root)
    p.write_text(original)
    corpus = root / "corpora" / f"{CORPUS}.jsonl"
    corpus.unlink()
    corpus.symlink_to(BENCH / "corpora" / f"{CORPUS}.jsonl")
    with pytest.raises(ValueError, match="escapes"):
        registry.load(root)


def test_row_validation_ids_labels_composition():
    entry = registry.load(BENCH)[CORPUS]
    rows = registry.rows(BENCH, CORPUS)
    for key, value in [("id", rows[1]["id"]), ("label", "safe"), ("dataset", "unknown")]:
        bad = copy.deepcopy(rows)
        bad[0][key] = value
        with pytest.raises(ValueError):
            registry.validate_rows(CORPUS, bad, entry)


def test_source_immutability():
    # Frozen files must match the checkout's committed baseline, even after tests.
    root = BENCH.parent
    paths = [
        "benchmarks/corpora",
        "recipes",
        "examples/inkling-2-mini-derisked",
        "benchmarks/protocol/SCORING.md",
        "benchmarks/protocol/responses.schema.json",
        "benchmarks/protocol/judgments.schema.json",
        "benchmarks/protocol/metrics.schema.json",
    ]
    proc = subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *paths],
        cwd=root,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0


def test_collection_output_mutation_prevents_completion(tmp_path, monkeypatch):
    a = args(tmp_path)
    calls = 0

    def transport(*unused):
        nonlocal calls
        calls += 1
        if calls == 2:
            a.out.write_bytes(b"corrupt")
        return good_raw()

    monkeypatch.setattr(collect, "chat_completion", transport)
    with pytest.raises(ValueError, match="changed during collection"):
        collect.collect(a)
    assert not Path(str(a.out) + ".complete.json").exists()


def test_empty_truncated_and_timeout_are_not_success(tmp_path, monkeypatch):
    a = args(tmp_path)
    cases = iter(
        [
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        ]
    )

    def transport(*unused):
        result = next(cases, None)
        if result is None:
            raise TimeoutError("secret endpoint detail")
        return result

    monkeypatch.setattr(collect, "chat_completion", transport)
    receipt = collect.collect(a)
    assert receipt["errors"] == 10
    rows = c.jsonl(a.out, "responses")
    _, jp = files(tmp_path, [], [judgment(r) for r in rows])
    result = s.score(a.corpus, a.out, jp, completion_path=Path(str(a.out) + ".complete.json"))
    assert result["states"] == {"ERROR": 10}
    assert result["coverage"]["collection_complete"]
    assert "secret endpoint" not in a.out.read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", float("nan")),
        ("top_p", float("inf")),
        ("max_tokens", 0),
        ("model_digest", "bad"),
        ("run_id", ""),
    ],
)
def test_invalid_settings_before_transport(tmp_path, monkeypatch, field, value):
    a = args(tmp_path)
    setattr(a, field, value)

    def forbidden(*unused):
        pytest.fail("transport must not run")

    monkeypatch.setattr(collect, "chat_completion", forbidden)
    with pytest.raises(ValueError):
        collect.collect(a)
    assert not a.out.exists()


def test_snapshot_hashes_match_parsed_bytes(tmp_path, monkeypatch):
    r = response()
    rp, jp = files(tmp_path, [r], [judgment(r)])
    expected = c.sha256(rp)
    original = c.jsonl_bytes
    changed = False

    def parse(data, *args, **kwargs):
        nonlocal changed
        result = original(data, *args, **kwargs)
        if not changed and args and args[0] == "responses":
            changed = True
            rp.write_text("changed after snapshot")
        return result

    monkeypatch.setattr(c, "jsonl_bytes", parse)
    result = s.score(CORPUS, rp, jp)
    assert result["responses_sha256"] == expected
    assert result["states"]["ENGAGED"] == 1


def test_completion_hash_mismatch_on_valid_json(tmp_path, monkeypatch):
    a = args(tmp_path)
    monkeypatch.setattr(collect, "chat_completion", lambda *unused: good_raw())
    collect.collect(a)
    records = c.jsonl(a.out, "responses")
    records[0]["answer_text"] = "modified but schema valid"
    a.out.write_bytes(b"".join(c.canonical(r) + b"\n" for r in records))
    _, jp = files(tmp_path, [], [judgment(r) for r in records])
    with pytest.raises(ValueError, match="completion receipt mismatch"):
        s.score(a.corpus, a.out, jp, completion_path=Path(str(a.out) + ".complete.json"))
