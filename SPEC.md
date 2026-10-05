# SPEC: arbiter++ evaluation middleware between models

Status: revision 4, 2026-09-14. arbiter++ is the upgrade's codename; the package and CLI remain
`arbiter`. The product is evaluation middleware between communicating models.
Revision 4 makes that role explicit, brings the basic middleware path into v1, and adds diagrams for
the product flow, replay bundle, verification, process boundaries and investigation loop.

Revision 3 addressed the specification review:

- Freeze verification thresholds and the complete execution manifest before evaluation.
- Separate untrusted REPL messages from authoritative events and enforce budgets outside the REPL.
- Measure interventions against paired unpatched baselines, with identical treatment on controls.
- Resolve patch positions afresh for each holdout using a recorded selector and tokenizer.
- Validate SAE model/site/preprocessing bindings and explicitly align activation device and dtype.
- Keep transcript detection fixtures and add a small model with a known causal mechanism.

Revision 1 was reviewed against Anthropic's automated-alignment-researcher (AAR), released
2026-08-28. The following revision-2 changes and dated upstream checks remain historical context;
revisions 3 and 4 do not recertify every historical upstream claim:

- The OS isolates untrusted agent code (rule 5, Section 5).
- A pre-execution code monitor and a post-hoc trajectory audit, lifted from AAR (Section 5.3).
- Findings are pre-registered, carry a control condition, and declare their holdout kind (Section 4.4).
- The M1 gate was corrected for bf16 precision and TransformerLens centering (Section 13).
- The loop shrinks to ~100 lines and moves to the other side of a pipe. Recursion is cut from v1.
- The existing planted-quirk fixtures in `results/` are kept (Section 12).
- Further factual corrections appear in Appendix B, items 12–17.

---

## 0. Overview

Arbiter++ sits at the handoff between models in a multi-model application. It captures their exchanges,
evaluates individual outputs and conversations against versioned criteria, and returns structured
scores, flags and supporting evidence. The application's messages continue through its existing
transport while evaluations run alongside them. The same evaluator can assess a recorded transcript
offline. When an evaluation calls for a deeper explanation, an investigation agent can run behavioral
or interpretability probes; causal claims then go through a separate verifier with fresh holdouts and
paired controls. Evaluation is the main product. Probes, artifacts and verification make its conclusions
inspectable and reproducible.

**Where it sits in an application.** Model A and Model B are participants in the application; the judge
and investigation agent are separate model roles used by arbiter++. Solid arrows carry application
messages. Dashed arrows carry captured data and evaluation work. Reverse messages use the same path.

```mermaid
flowchart LR
    A["Application model A"] -->|"message"| M["Arbiter++ middleware hook"]
    M -->|"forward through existing transport"| B["Application model B"]
    M -.->|"capture exchange and session context"| H["Evaluation controller"]
    R["Versioned rubric and evaluator"] -.-> H
    H -.-> E["Evaluation: scores, flags, evidence"]
    E -.->|"configured escalation"| I["Investigation and probes"]
    I -.-> F["Proposed finding"]
    F -.-> V["Independent verification"]
    V -.-> O["Recorded outcome and trial evidence"]
    E -.-> C["Application or researcher"]
    O -.-> C
```

AAR uses a generic coding agent with Bash, file tools and seven MCP tools, with one fresh session per
iteration. It beat 28 human researchers. Its environment isolates held-out data at the OS level, checks
every method with a code monitor before execution, runs held-out evaluations with a capability gate,
and provides a shared forum. In AAR's environment ablation, removing literature review or the forum
reduced performance; removing internet access had no detectable effect. Arbiter uses OS isolation, code
review, held-out evaluation and shared research files, with an agent loop of ~100 lines.

**The research handoff.** AAR proposes training methods, trains and scores them. Arbiter++ evaluates
model interactions and can investigate why a behavior occurred. A diagnosis can hand off an artifact:
a verified steering direction (`ArtifactRef(kind="vector")`) is an input to an AAR method. AAR's own paper found
activation steering in 24% of all methods and 74% of unconstrained sycophancy methods, always to build a
better teacher.

---

## 1. What arbiter becomes

An evaluation layer for model-to-model exchanges and complete conversations, with a reproducible
investigation process behind it. It works on hosted text-only models as well as models whose internals
are accessible. A rubric can assess alignment, deception, collusion, coherence, or task-specific quality;
the run records which criteria, evaluator and context produced each score.

Its public records start with an **exchange** (who sent what to whom in which session) and an
**evaluation** (the rubric, scores, flags and evidence for selected exchanges). Investigation adds the
**subject handle** (one model connection with declared capabilities), the **artifact** (a tensor with its
provenance), and the **finding** (a claim and a runnable check). A completed evaluation is an evaluator's
assessment. A verified finding has passed the separate reproduction protocol; these are distinct outputs.

Interpretability needs a tensor type in the public contract. The seven agent frameworks studied record
text, tokens and at best logprobs; none has a tensor type. Stringifying tensors into a `details` field
loses the structure that probes need. Arbiter defines the artifact type and subject handle first, then
builds the loop, record and plugin contract around them.

AAR's post-hoc audit found cheating in 2.4% of 1,601 trajectories, even though its agents were not trying
hard to evade detection. Arbiter's agent may run for hours with a REPL. The design treats it as untrusted
and assumes it may try to make a finding look verified.

### What a session looks like

```text
$ arbiter investigate \
    --role subject=local/google/gemma-2-2b \
    --role agent=openai-compatible/deepseek/deepseek-v3.2 \
    --transcript taps/2026-08-25.jsonl \
    "why does the subject recommend the same product no matter the question?"
```

The transcript lands as a variable in the REPL, never in the agent's context. A middle turn of the loop
looks like this. The fenced code runs in the REPL; subject and registered probe calls execute in the
subject worker through a recording proxy. Each call and result becomes a JSONL Event:

```python
s    = Site.parse("resid_post@11")
acts = sub.read(s, convo[41].text)[s]  # <Acts resid_post@11 (1, 57, 2304) bfloat16 raw_hf 9f3c1ab2e4d1>
top_features(sub, site=str(s), text=convo[41].text,
             release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_82")
```

The probe call refuses: `site='resid_post@11' (SAE requires 'resid_post@12')`, with the fix named in the
traceback. The agent changes `s` to `Site.parse("resid_post@12")` and re-runs. The model revision and
preprocessing must also match the SAE's checked binding; a backend name alone is not that guarantee.
The run ends with a finding, and a separate verifier checks it later:

```text
$ arbiter verify runs/2026-08-26T14-02/f_003.json
f_003  "feature 7134 fires on product mentions; zeroing resid_post@12 there flips the recommendation"
       effect   8/8 paired intervention deltas match    [wilson 95%: 0.68–1.00]
       control  8/8 paired deltas remain zero            [wilson 95%: 0.68–1.00]
       → verified:scenario   (every trial in verify_runs; domain/format holdouts not run)
```

Everything above replays from the JSONL and its content-addressed execution bundle (Section 4.4).

---

## 2. Design rules

1. **Give tensors a public type.** Activations carry provenance through `ArtifactRef` and `Acts`.
2. **Refuse, never degrade.** A probe that needs `resid_post@12` against a hosted model is a loud
   registration error. It must never return a silently empty result.
3. **Every refusal names its own fix** in the exception text. The agent repairs itself from the traceback.
4. **A claim without a runnable check cannot be constructed.** `Finding.check` is required.
5. **The agent is untrusted, and the boundary is the OS.** The REPL cannot write the run record, artifact
   index, model weights, or worker code, or read fixture answer keys or provider credentials. Its IPC
   messages are untrusted requests. It cannot launch a privileged verifier or subject worker.
6. **Model-visible implies logged.** Anything that reached the agent's context is a JSONL line. That
   includes thinking blocks when the provider returns them.
7. **Pre-register.** Hash the complete proposal, including `n`, `k`, selectors, and execution manifest,
   before verify runs. Editing any of it makes a new finding.
8. **Every effect ships with a control and a baseline.** Verification measures treatment-minus-baseline
   on both effect and control stimuli. A causal claim without this comparison is `unfalsifiable`.
9. **Synchronous core.** A researcher never writes `async def` in a probe.
10. **`import arbiter` must not import torch.** Enforced by a test.
11. **Plain functions over base classes.** A probe is a decorated function in a `.py` file you own.
12. **Keep the loop small.** Target ~100 provider-agnostic lines. Probes, storage and verification live
    outside it.
13. **Build the eval before the agent.** The M3 fixtures must pass before the agent loop ships.
14. **Remove the old sweep product.** Preserve its reusable code and fixtures as specified in Section 12.
15. **Budgets survive hostile code.** The controller meters model requests and supervises execution limits
    outside the REPL. A Python loop cannot reset a counter or extend its own deadline.

---

## 3. Model access and backends

| Tier | What it is | Examples | Needs |
|---|---|---|---|
| **W** white-box | Live `nn.Module` in the probe's subject worker, on its GPU | nnterp, TransformerLens, SAELens, captum | Weights you host |
| **G** grey-box | Real internals over HTTP against weights you do not have | nnsight `remote=True`/NDIF, Neuronpedia | An account |
| **B** black-box | Text in, text out | judge, elicitation, prefill, Petri/Bloom transcripts | Any API |

Tier W has two incompatible model-object contracts:

- **Hook-name-string**: TransformerLens, SAELens, circuit-tracer.
- **Module-proxy**: nnsight, nnterp, pyvene.

nnterp/nnsight is the primary in-process substrate. TransformerLens is a second backend behind the same
`Site` vocabulary, preserving access to SAELens and circuit-tracer.

These versions cannot share a virtualenv. Verified 2026-09-01:

```
TransformerLens 3.8.1   requires transformers>=5.9.0
circuit-tracer (git)    pins nnsight==0.6.1, transformers>=4.56.0,<=4.57.3
current nnsight         0.7.0
```

The conflicting pins require separate processes, which exchange safetensors files and provenance through
`ArtifactRef`. A probe with incompatible pins lives in its own locked venv, is invoked as
`uv run --frozen`, and hands back an `ArtifactRef`. Dependency isolation and trust isolation are separate:
approved probes run with the subject; arbitrary agent code runs in the REPL (Section 5.1).
The PyPI package named `circuit-tracer` is an unofficial third-party fork with looser pins. Pin the
official project's git source (Section 11).

Inspect AI's `transformer_lens.py` provider hosts TransformerLens behind a
uniform `ModelAPI`, takes a prebuilt model via `model_args["tl_model"]`, then discards every hook and
cache, preventing probes from using its internals. The same project's `nnterp.py` exposes internals
through a declared, typed, opt-in side channel.

---

## 4. Core contracts

Do not import torch at module level.
The code blocks specify public shapes and selected checks. The accompanying validation rules are
normative at both Python entrypoints and IPC ingestion; type annotations do not replace them.

### 4.1 Capability negotiation

```python
# arbiter/contracts.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TYPE_CHECKING, runtime_checkable
if TYPE_CHECKING:
    import torch

# Point is an open set that each backend advertises via Provides.points.
# The Literal is documentation and IDE completion. Provides.points is the enforcement.
Point  = Literal["embed", "resid_pre", "resid_post", "attn_out", "attn_z", "mlp_out", "logits"]
Cap    = Literal["generate", "logprobs", "raw_cot", "read_acts", "write_acts", "grads"]
Coords = Literal["raw_hf", "folded_ln", "none"]


@dataclass(frozen=True, slots=True)
class Site:
    """A symbolic, architecture-independent address into a model's internals.
    The same string works on gpt2, llama and gemma. `resid_post@12`, `attn_z@5.h7`."""
    point: Point
    layer: int | None = None
    head: int | None = None

    def __str__(self) -> str:
        s = self.point if self.layer is None else f"{self.point}@{self.layer}"
        return s if self.head is None else f"{s}.h{self.head}"

    @classmethod
    def parse(cls, s: str) -> "Site":
        point, _, rest = s.partition("@")
        layer, _, head = rest.partition(".h")
        return cls(point, int(layer) if layer else None, int(head) if head else None)


@dataclass(frozen=True, slots=True)
class Provides:
    """A Subject's capabilities, declared once at connect time."""
    model_id: str            # including @revision
    caps: frozenset[Cap]
    points: frozenset[Point] = frozenset()
    n_layers: int = 0
    coords: Coords = "none"
    preprocessing_id: str = ""  # digest of exact weight processing and activation transforms


@dataclass(frozen=True, slots=True)
class Needs:
    """What a probe requires. `coords=None` means it does not care."""
    caps: frozenset[Cap] = frozenset()
    points: frozenset[Point] = frozenset()
    coords: Coords | None = None
    min_layers: int = 0

    def unmet(self, p: Provides) -> list[str]:
        """Empty list == compatible. Anything else is a loud registration failure,
        and this list is also shown to the agent as 'unavailable because...'."""
        m  = [f"cap {c!r}" for c in sorted(self.caps - p.caps)]
        m += [f"site {s!r}" for s in sorted(self.points - p.points)]
        if self.coords is not None and p.coords != self.coords:
            m.append(f"coords={self.coords!r} (subject is {p.coords!r})")
        if p.n_layers < self.min_layers:
            m.append(f"n_layers>={self.min_layers} (subject has {p.n_layers})")
        return m
```

`Needs.unmet(Provides)` decides static capability mismatch at registration. Compatibility that depends
on arguments is also checked at invocation: an SAE's exact model, site, and preprocessing cannot be
inferred from a generic `read_acts` capability. Section 6 defines those checks.

### 4.2 The artifact and activation types

```python
@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Tensor metadata stored in Events, Results and Findings.
    The tensor itself stays on disk."""
    id: str                                  # "blake2b:9f3c..." content address
    kind: Literal["acts", "sae", "probe", "vector", "graph", "file"]
    shape: tuple[int, ...]
    dtype: str
    site: Site | None = None
    coords: Coords = "none"                  # activation coordinate space
    model_id: str = ""
    preprocessing_id: str = ""
    token_ids: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def address(self) -> str:                # Neuronpedia-style, human-quotable
        return f"{self.model_id}/{self.site}/{self.id[:12]}"


class Acts:
    """A named activation tensor plus its provenance. Loaded lazily from the store,
    with CPU loading by default. GPU copies have explicit owners and lifetimes.
    Its repr is safe to print."""
    __slots__ = ("ref", "_store", "_t")

    def __init__(self, ref: ArtifactRef, store: "Store", t: "torch.Tensor | None" = None):
        self.ref, self._store, self._t = ref, store, t

    def tensor(self) -> "torch.Tensor":
        if self._t is None:
            self._t = self._store.load(self.ref)
        return self._t

    def __repr__(self) -> str:               # the only tensor projection an agent sees
        return (f"<Acts {self.ref.site} {tuple(self.ref.shape)} {self.ref.dtype} "
                f"{self.ref.coords} {self.ref.id[:12]}>")


@dataclass(frozen=True, slots=True)
class PositionSelector:
    """Resolved against each stimulus's model input tokens."""
    kind: Literal["all", "last", "literal_span", "indices"] = "all"
    text: str = ""                            # literal_span: exact text in the stimulus
    occurrence: int = 0                       # zero-based occurrence; never fuzzy matching
    indices: tuple[int, ...] = ()             # indices: evidence only, not holdout verification


@dataclass(frozen=True, slots=True)
class Patch:
    """A JSON-serializable intervention the agent can emit and the recorder can
    log, diff and replay."""
    site: Site
    op: Literal["set", "add", "scale", "zero", "swap"]
    positions: PositionSelector = field(default_factory=PositionSelector)
    value: str | None = None                   # ArtifactRef.id of the tensor to write
    scalar: float | None = None


ResultKind = Literal["scalar", "per_token", "features", "graph", "text", "acts"]


@dataclass(frozen=True, slots=True)
class Result:
    """What a probe returns and what Finding.expect stores. A tagged union by `kind`.
    JSON-serializable end to end: tensors never appear inline, only as ArtifactRefs.
    kind == "acts" is what a bare Subject.read() emits through the recording proxy;
    it is evidence, and it is never a legal `expect`."""
    kind: ResultKind
    value: float | None = None                       # scalar
    per_token: tuple[float, ...] = ()                # per_token
    features: tuple[tuple[int, float], ...] = ()     # features: (index, activation)
    text: str = ""                                   # text
    source: str = ""                                 # human-quotable origin
    artifacts: tuple[ArtifactRef, ...] = ()          # graph and acts live here as refs
```

**Tolerance by result kind.** Verify uses the following comparators:

| kind | `matches(got, expect, tol)` |
|---|---|
| `scalar` | `abs(got - expect) <= tol * max(1.0, abs(expect))` |
| `per_token` | same length, and max abs diff satisfies the scalar rule |
| `features` | Jaccard of the top-k index sets `>= 1 - tol` |
| `text` | equal after whitespace/case normalization; `tol` ignored |
| `graph` | Jaccard of edge sets `>= 1 - tol` |
| `acts` | never an `expect`; refuse at `Finding` construction |

**Coordinate compatibility.** Matching `coords` does not establish that the model revision, layer,
centering flags, or activation normalization match. `preprocessing_id` identifies
the exact transforms in the execution manifest; SAE and intervention compatibility checks compare that
identity as well as the model and site. Unknown SAE preprocessing is a refusal until a checked binding
exists. Do not infer an SAE's training coordinates from its use of SAELens or TransformerLens.

**Dtype policy (required).** Hash and store via a bf16-safe byte reinterpret, `t.view(torch.uint8)`, or
safetensors' native bf16. Never `.numpy()`, which raises on bfloat16. The torch-free ingest path uses
safetensors' numpy backend, which has no bf16, and must refuse bf16 loudly rather than misread it.

### 4.3 The subject handle

```python
@dataclass(slots=True)
class Turn:
    """One model response. `cot=None` means the provider refused to supply it
    (Anthropic always does for raw CoT), not that the model did not think.
    `thinking` is the provider's summarized/visible thinking, when any."""
    text: str
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float, ...] | None = None
    cot: str | None = None
    thinking: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class Subject(Protocol):
    """The model handle passed to every probe, with one owner and one lifetime."""
    provides: Provides
    device: str
    manifest_id: str

    def generate(self, msgs: list[dict[str, str]], **kw: Any) -> Turn: ...

    def read(self, sites: Site | list[Site], text: str | list[str], *,
             positions: slice | int = slice(None),
             logits: bool = False,
             keep: Literal["cpu", "gpu"] = "cpu") -> dict[Site, Acts]:
        """Requires cap 'read_acts'. Nothing is captured that was not asked for.
        Writes every tensor to the store and returns handles. Raises Refused(site)
        rather than returning zeros."""

    def run(self, msgs, *, edits: tuple[Patch, ...] = (), sites: tuple[Site, ...] = ()
            ) -> tuple[Turn, dict[Site, Acts]]:
        """Requires cap 'write_acts' when edits is non-empty. An intervention is an
        argument to a single call, never a context manager the agent holds open across
        turns. Interventions demonstrably leak (repeng.set_control persists,
        TL's is_permanent=True survives remove_all_hook_fns, enable_compatibility_mode
        mutates weights with no undo). Teardown is asserted by hook count in `finally`."""

    def close(self) -> None: ...
```

`Subject` exposes the same interface across all tiers, including G and B, where the weights are behind
HTTP. In the REPL it is an RPC proxy. The trusted subject worker supplies the recording proxy
(Section 5.2): every verb emits a `Call` + `Result` pair, with `probe="subject.read"`, `"subject.run"` or
`"subject.generate"`. REPL artifact handles load CPU copies from read-only committed blobs.

### 4.4 The record, and what "verified" means

```python
@dataclass(frozen=True, slots=True)
class ExecutionManifest:
    """Canonical JSON stored with the run; id hashes the body excluding id."""
    id: str
    schema_version: int
    roles: dict[str, dict[str, Any]]           # resolved connection and generation settings
    participants: dict[str, dict[str, Any]]    # application endpoint ids -> resolved model identities
    probes: dict[str, dict[str, Any]]          # registry name -> entry symbol, source bundle, lock
    artifacts: dict[str, ArtifactRef]          # all input tensors/configs, including SAE weights
    runtime: dict[str, Any]                    # Python/packages, device, dtype, deterministic settings
    policies: dict[str, Any]                   # rubrics, context selection, comparison, limits, holdouts


@dataclass(frozen=True, slots=True)
class Call:
    """A JSON-serializable probe invocation used as the replay unit."""
    probe: str                              # registry name, or "subject.<verb>"
    kwargs: dict[str, Any]
    manifest_id: str                        # exact execution environment, never a mutable alias
    role: str = "subject"
    stimulus: str | None = None             # the kwarg verify swaps for holdout inputs
    patches: tuple[Patch, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> ArtifactRef.id
    seed: int | None = None


EventKind = Literal[
    "run.start", "turn.code", "turn.thinking", "turn.stdout", "turn.subcall",
    "monitor", "call", "result", "exchange", "evaluation", "finding", "verify", "audit",
    "error", "tap.turn", "drop",
]


@dataclass(frozen=True, slots=True)
class Event:
    """One append-only JSONL line in the canonical storage format.
    Only the controller writes Events; child requests are untrusted payloads."""
    run: str
    seq: int
    ts: float
    kind: EventKind
    data: dict[str, Any]
    parent: int | None = None


@dataclass(frozen=True, slots=True)
class Exchange:
    """Application traffic captured by the trusted integration adapter."""
    id: str
    session_id: str
    sender: str                              # participant id in the manifest
    recipient: str                           # participant id in the manifest
    content: str
    manifest_id: str
    parent_id: str | None = None              # correlation for replies / model handoffs


@dataclass(frozen=True, slots=True)
class Evaluation:
    """A rubric assessment of exchanges; verification is recorded on Findings."""
    id: str
    exchange_ids: tuple[str, ...]             # exact context window assessed
    manifest_id: str
    rubric_id: str                           # digest of criteria, scales and decision thresholds
    status: Literal["completed", "error", "dropped"]
    evaluator_call: Call | None = None        # required for completed evaluations
    scores: dict[str, float] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    rationale: str = ""
    evidence_events: tuple[int, ...] = ()      # seq values in this run
    finding_ids: tuple[str, ...] = ()          # linked investigations; never an implied verified status


HoldoutKind = Literal["scenario", "domain", "format"]


@dataclass(frozen=True, slots=True)
class HoldoutCase:
    id: str
    effect: Any                              # JSON value matching the stimulus kwarg's schema
    control: Any = None                       # separate negative stimulus for intervention claims


@dataclass(slots=True)
class Finding:
    """A claim about the subject with a runnable check and, for interventions,
    a control that tests whether the effect is specific."""
    id: str
    claim: str
    subject: str                              # model_id@revision
    sites: tuple[Site, ...]
    evidence: tuple[ArtifactRef, ...]
    check: Call                               # re-run this. Data, never a code string.
    expect: Result                            # intervention: expected signed scalar delta
    kind: Literal["observation", "intervention"] = "observation"
    control: Call | None = None               # same probe, inputs where the effect must not appear
    control_expect: Result | None = None
    tol: float = 0.05
    holdout_kind: HoldoutKind = "scenario"    # what the agent claims it generalizes over
    holdout: tuple[HoldoutCase, ...] = ()      # filled in by verify, never by the agent
    n: int = 8                                # == len(holdout) once verify has run
    k: int = 8                                # trials that must reproduce; k == n means unanimous
    proposed_hash: str = ""                   # set by the recorder at the `finding` event
    provenance: Literal["investigation", "tap"] = "investigation"
    status: Literal["proposed", "verified", "refuted",
                    "unreproducible", "unfalsifiable"] = "proposed"
    verified_on: tuple[HoldoutKind, ...] = () # which holdout kinds passed
    verify_runs: tuple[dict, ...] = ()        # every effect and control trial
    verified_by: str = ""                     # run id of the verify pass; never set by the agent

    def __post_init__(self):
        if type(self.n) is not int or type(self.k) is not int or not 1 <= self.k <= self.n:
            raise ValueError("Finding requires integers 1 <= k <= n; set the trial policy before proposal")
        if self.expect.kind == "acts" or (self.control_expect and self.control_expect.kind == "acts"):
            raise ValueError("Finding.expect cannot be activations; choose a measured Result")
```

The controller assigns Exchange and Evaluation IDs, source identity and status from authenticated
adapter/evaluator requests. Completed evaluations require an evaluator call, the exact input context,
and scores/flags that validate against the frozen rubric. Errors and drops carry a reason and no
success score. Evaluator text and participant content are data, never authority to change run status.

An observational finding is verified when its recorded check reproduces `expect` within `tol` in at
least k of n fresh holdout cases. An intervention finding is verified when at least k of n paired cases
pass both the effect-delta and control-delta checks below. Replay uses the frozen execution bundle,
fresh workers, and no result cache. An observation does not support causal wording such as "flips" or
"causes"; causal findings must use the intervention contract.
Observations contain no patches. If an observation supplies a control, it must also supply
`control_expect`, and both absolute-result comparisons must pass in the same k cases.

**One intervention case means four measurements.** Each arm uses a fresh worker with the same frozen
model and generation settings. The verifier computes the two deltas from the returned measurements.

```mermaid
flowchart TB
    E["Effect stimulus"] --> EB["Unpatched baseline"]
    E --> ET["Treatment with recorded patch"]
    C["Negative control stimulus"] --> CB["Unpatched baseline"]
    C --> CT["Treatment with the same patch and selector"]
    EB --> ED["Effect delta = treatment - baseline"]
    ET --> ED
    CB --> CD["Control delta = treatment - baseline"]
    CT --> CD
    ED --> EP["Matches the nonzero expected effect?"]
    CD --> CP["Matches zero within tolerance?"]
    EP --> P["Case passes only when both checks pass"]
    CP --> P
    P --> N["At least k of n paired cases must pass"]
```

**Finding outcomes.** Malformed proposals are rejected before creating a finding. Missing causal
requirements produce `unfalsifiable`; missing replay inputs or failed execution produce `unreproducible`.
Changing a frozen claim or its thresholds creates a new proposal. An audit adds evidence without
rewriting an outcome.

```mermaid
stateDiagram-v2
    state "proposed" as Proposed
    state "unfalsifiable" as Unfalsifiable
    state "unreproducible" as Unreproducible
    state "refuted" as Refuted
    state "verified" as Verified
    [*] --> Proposed: Valid frozen proposal
    [*] --> Unfalsifiable: Missing checkable causal requirements
    Proposed --> Unreproducible: Bundle, holdout or execution failure
    Proposed --> Refuted: Valid trials fall below k of n
    Proposed --> Verified: Replay gates and k of n comparisons pass
    note right of Verified: Applies only to the holdout kinds tested
```

Verification requirements:

- **From the execution bundle.** JSONL records plus their manifest and referenced blobs must be
  sufficient. Missing or hash-mismatched inputs mean `unreproducible`; never fall back to current files,
  current registry discovery, or a newly downloaded `main` revision.
- **Fresh workers.** Each baseline and treatment arm starts from the same frozen model state with the
  same seed and generation settings, isolated from other arms and the investigation. An implementation
  may reuse a worker only after a reset with equivalent state isolation has its own regression gate.
- **Holdout.** `verify` generates and records the holdout; it ignores agent-authored cases.
  Generate and freeze all n cases before executing any arm. Every trial needs a distinct stimulus.
  Each intervention case has an effect stimulus and a separate negative control stimulus.
  `n == len(holdout)`; neither n nor k is rewritten after failures.
  Reject duplicate rendered effect stimuli before execution and require n distinct complete cases.
  Record duplicates, generation failures, and missing cases as an `unreproducible` pass; they cannot be
  silently replaced after seeing outcomes or removed from the denominator.
- **Three holdout kinds, from AAR's benchmark design.** *scenario*: paraphrases of the stimulus.
  *domain*: other prompts from the fixture set tagged with the same behavior. *format*: the same content
  in a different wrapper, such as system vs user turn, multi-turn, raw completion. v1 auto-generates
  scenario only; domain and format require a fixture set (Section 13, M3). The status line says which
  ran: `verified:scenario` covers fewer conditions than `verified:scenario,domain`. In AAR's ablation,
  hill-climbing on one benchmark produced zero generalization on unseen ones.
- **Paired control.** v1 intervention checks use an approved scalar metric probe. For each case the
  verifier derives an unpatched baseline by clearing `Call.patches`, then runs the treatment call on
  the identical input. It computes `delta = treated.value - baseline.value` itself. Effect delta must
  match `expect`; control delta must match `control_expect`, which must be scalar zero. Both must pass
  in the same case. `verify_runs` retains all four raw Results, both deltas, seeds, and comparisons.
  A flip claim uses a frozen binary/thresholded metric so its expected delta denotes a flip.
  `matches(0, expect, tol)` must be false; a no-op cannot qualify as a measured effect.
- **Same treatment.** `check` and `control` must have identical probe, manifest, role, seed, artifact
  bindings, selector, patch operations and values, and non-stimulus kwargs. Only the stimulus value
  differs. The metric probe cannot install hidden interventions through kwargs or global state; its
  approved implementation receives the recorder's `Call.patches` as the sole treatment input. No
  control, a changed treatment, an absolute-output expectation, or a non-scalar causal metric means
  `unfalsifiable`. A call with patches cannot be submitted as an observational finding.
- **Every trial recorded, with an interval.** `verify_runs` holds every effect trial and every control
  trial. v1 prints the Wilson 95% interval on the pass rate without using it as a gate. With n=8 the
  interval is wide.
- **Pre-registered.** `proposed_hash` is blake2b over canonical JSON of `(schema_version, kind, claim,
  subject, sites, evidence, check, expect, control, control_expect, tol, holdout_kind, n, k, provenance)`.
  The schema version is in the execution manifest; `check` and `control` bind that manifest and the
  complete selectors and artifact references. The recorder validates and deep-copies this payload,
  including finite nonnegative tolerance, finite result values, and `1 <= k <= n`, before hashing it.
  Dataclass annotations alone do not validate untrusted JSON. The `verify` event references the hash;
  any proposal-field change requires a new finding and a fresh evaluation. Verification-derived fields
  (`holdout`, status, trial records, verified kinds and run id) never enter the proposal hash.

**Execution manifest.** Before the first call, the controller captures and hashes:

- Probe entry symbols, their complete source/resource bundles and transitive imports, exact package
  versions and lockfiles, and the arbiter code/schema revision. Snapshot approved filesystem plugins;
  replay never imports the mutable discovery directories. Undeclared file/network dependencies refuse.
- Every role's resolved backend, model commit, tokenizer revision, chat template and special-token
  policy, adapters, dtype/quantization, weight-processing flags, activation transforms, generation
  defaults, and provider identity. Record hardware/kernels and random/determinism settings too.
- Input artifact bytes and provenance, SAE weights/config/bindings, selector implementation, comparator
  version, evaluation rubrics and context/sampling policy, holdout-generation policy/model/prompt, and
  resource limits. Resolve defaults explicitly. Freeze application participant identities separately
  from the judge and investigation roles.

The manifest body and referenced blobs are immutable, verified on load, and retained while a finding
references them. A digest of tensor bytes alone deduplicates storage; it does not identify model/site
provenance. Credentials are injected privately by the controller and never stored in the manifest. A
hosted role without an immutable model version records its fingerprint and is marked best-effort;
`verified` requires an attestable matching version, otherwise the outcome is `unreproducible`.
Reconnects, new approved plugins, or new committed inputs produce a new manifest before use. The
recorder normalizes calls to explicit kwargs and checks their roles/artifacts against that manifest.
Run-level budget counters remain cumulative across manifest changes.

**What must travel with a replay.** Names such as a probe name or model alias resolve through the
manifest. The referenced bytes and settings are part of the execution bundle.

```mermaid
flowchart TB
    P["Approved probe source and locked dependencies"] --> M["Immutable execution manifest"]
    S["Participants, role models, tokenizer and preprocessing"] --> M
    A["Artifact bytes, provenance and SAE bindings"] --> M
    R["Rubric, comparison rules, selectors and limits"] --> M
    M -->|"manifest digest"| C["Normalized Call"]
    C --> E["Evaluation or proposed Finding"]
    T["Finding expectations, controls, n and k"] --> F["Proposal hash"]
    E -->|"when proposing a finding"| F
    M --> B["Replay bundle"]
    C --> B
    F --> B
    J["JSONL events and referenced blobs"] --> B
    B --> V["Fresh replay workers"]
```

**Cache key** is blake2b over canonical JSON of `(manifest_id, normalized_Call, resolved_input_tokens,
resolved_patch_positions, effective_seed)`. Canonical JSON sorts keys, rejects NaN/Infinity, and has a
versioned encoding for `Site`, selectors, tuples and slices. This separates raw and processed models,
different probe implementations, and different tokenization even when the model revision is unchanged.
Cache lookup still checks blob integrity. Verification disables result caching for every arm.

**Holdout alignment.** `stimulus` names one kwarg with a declared JSON schema (text, or structured
messages for format holdouts). `PositionSelector` is interpreted by trusted worker code over the exact
tokenized model input, using the tokenizer/template pinned in the manifest. v1 patches affect prompt
prefill only: `all` selects all non-padding prompt tokens, `last` selects the last one, and
`literal_span` selects tokens overlapping the specified occurrence's character span in the stimulus.
Structured wrappers must preserve a recorded mapping into rendered model-input offsets. Missing or
ambiguous spans and unsupported offset mappings refuse; the verifier never guesses a token position.
Explicit `indices` are allowed for exploration but make a holdout intervention `unfalsifiable`.

Store each arm's rendered input, token IDs, attention mask, resolved positions and selector hash.
Resolve the same selector separately on effect and control inputs; baseline and treatment within each
pair share identical tokens and position resolution. A v1 holdout patch may broadcast a site-shaped
vector over selected tokens; an artifact with a donor sequence/batch axis cannot be reused on a new
stimulus. Such swaps remain exploratory until a recorded donor-alignment contract exists. Check shape,
model, site and preprocessing before execution. An alignment/execution failure is recorded against the
planned case and makes the pass `unreproducible`; a valid measured mismatch counts as a failed trial
and yields `refuted` if fewer than k paired cases pass. Never reduce n or silently skip invalid cases.

---

## 5. The loop, the pipe, and the monitor

### 5.1 Four process roles

The evaluation adapter feeds the trusted controller. Arbitrary Python, model execution and private
holdouts live under different OS identities. All model requests and authoritative records cross the
controller; there is no direct REPL-to-worker or REPL-to-verifier connection.

```mermaid
flowchart TB
    A["Trusted middleware adapter"] -->|"captured exchanges"| H
    subgraph Harness["Controller process: arbiter"]
        H["Broker, recorder, monitor and evaluator orchestration"]
        J[("JSONL run record")]
        S[("Manifest, artifact store and index")]
        H -->|"sole writer"| J
        H -->|"validate and commit"| S
    end
    R["REPL: arbiter-repl<br/>Untrusted Python and bounded scratch"]
    W["Subject worker: arbiter-subject<br/>Weights and approved probes; no fixture keys"]
    V["Verifier: arbiter-eval<br/>Holdouts and paired comparisons"]
    K[("Private fixture keys")]
    H -->|"approved exec and close"| R
    R -->|"typed requests, stdout and proposals"| H
    H -->|"budgeted model and probe calls"| W
    W -->|"measurements and staged blobs"| H
    V -->|"fresh trial requests and reports"| H
    H -->|"worker measurements"| V
    K -->|"evaluator-only access"| V
    S -.->|"read-only committed artifacts"| R
```

The REPL exposes `exec(request_id, code) -> (stdout_prefix, stdout_len, proposals)` and `close()` on a
private inherited socket. During execution its proxy sends typed subject/probe requests to the controller;
the controller services those while draining output, rather than waiting for `exec` to finish. Only the
controller can address the subject worker's socket. The worker exposes registered calls and close, never
arbitrary `exec`. Approved probes remain ordinary Python functions and run beside the model; the REPL
catalog binds stubs for them. A new or changed plugin needs researcher approval and a new manifest before
it can execute there. Other agent-written calculations run in the REPL on copied artifacts.

The REPL cannot access GPU devices, worker memory, model caches, provider credentials, fixture keys, or
the network. It gets a bounded scratch filesystem and read-only committed artifacts. The worker cannot
read keys or write `runs/` or the artifact index; the controller validates and commits its staged blobs.
API calls route through the controller's metered credential gateway. Apply distinct users plus filesystem,
process, device and network isolation at launch; a same-user `chmod` is not an accepted substitute.
Fail closed if these controls cannot be established. An explicitly selected local development mode may
relax them, but records `isolation=unsafe` and cannot emit verified findings.

**Message authority.** The transport authenticates peers using OS identity/private inherited endpoints,
not a claimed `source` field. No child may submit an arbitrary `Event`. The recorder enforces:

| Sender | Accepted payloads | Recorder action |
|---|---|---|
| Application integration adapter | exchange envelope with session, participant and parent identifiers | Authenticate the adapter; assign exchange identity and validate participant mapping |
| REPL | subject/probe requests, bounded stdout, artifact candidates, finding proposals, error reports | Validate schema, request id and quotas; label output/proposals as agent-originated |
| Subject worker | measurement results and staged artifacts for an admitted call | Match outstanding request; validate results and blob digests; record worker origin |
| Verifier | trial/comparison reports for a known proposal hash and manifest | Validate all n cases and recompute the paired pass count before recording status |
| Controller / evaluator / monitor / audit | run metadata, evaluations, code, thinking, admission, approvals, audit results, drops | Emit through dedicated trusted paths |

The controller supplies `run`, `seq`, `ts`, `parent`, finding IDs, proposal hashes, provenance and initial
status. Proposal payloads cannot set `holdout`, `verified_on`, `verify_runs`, `verified_by` or status.
Reject unknown/reserved fields and child-originated `verify`, `monitor`, or `audit` event requests.
The verifier has a separate endpoint inaccessible to both REPL and worker. Forged measurements printed
by the agent remain stdout.

Each dependency environment is a subject worker with the same typed call protocol and a frozen lock.
The persistent REPL survives ordinary probe calls without sharing the worker's Python namespace.

### 5.2 The loop

The loop targets ~100 lines:

1. Bind the transcript under investigation to a variable in the REPL's namespace. Never paste it
   into context.
2. The agent sees constant-size metadata: length, a short prefix, how to address parts of it, and the
   probe catalog.
3. It emits a fenced Python block. The controller runs the monitor (5.3), then sends the block to the
   server. Stdout comes back as a bounded prefix plus its length. Truncation keeps a 200k-token
   activation dump out of the model's context; the full data stays in REPL variables.
4. Terminate on `FINAL(value)` or `FINAL_VAR(name)`. Validate the two explicitly; the RLM paper's
   distillation data had 16% of turns misuse one and 13% misuse the other.
5. The controller charges agent turns and tokens; its broker reserves subject-call/token budget before
   each request and reconciles measured usage afterwards. Per-probe code does not implement accounting.

**One investigation turn.** The monitor and recorder are parts of the controller. A refusal returns an
error the agent can use on its next turn; it does not execute the rejected code or model request.

```mermaid
sequenceDiagram
    participant A as Investigation agent model
    participant H as Controller and monitor
    participant R as Untrusted REPL
    participant W as Subject worker
    participant L as JSONL record
    H->>A: Metadata, catalog and bounded prior output
    A-->>H: Code and visible thinking
    H->>L: Record code, thinking and monitor decision
    alt Monitor approves exact request, code and manifest
        H->>R: exec(request_id, code)
        loop Each model or registered probe request
            R->>H: Typed call request
            alt Schema, compatibility and budget allow the call
                H->>L: Record admitted Call
                H->>W: Execute from frozen bundle
                W-->>H: Result and staged artifacts
                H->>L: Record Result and committed artifact refs
                H-->>R: Result handles
            else Request refused
                H->>L: Record refusal
                H-->>R: Error with repair guidance
            end
        end
        R-->>H: Bounded stdout and proposal payloads
        H->>L: Validate and record output or proposed finding
    else Monitor refuses
        H-->>A: Refusal and repair guidance
    end
    Note over H,W: Controller enforces deadlines and quotas throughout execution
```

**Execution limits.** The resolved configuration must include finite `exec_timeout_ms`,
`call_timeout_ms`, `run_timeout_ms`, `max_subject_calls`, `max_model_tokens`, `max_stdout_bytes`,
`max_ipc_bytes`, `max_artifact_bytes`, `max_scratch_bytes`, and `max_memory_bytes`. Record these in the
manifest and `arbiter dump-config`; a plugin timeout may tighten, never extend, the run's limits.

- Deadlines are measured by the controller, which can terminate the complete REPL/worker process group.
  The agent has no access to the counters or launch handles. Enforce memory, scratch and device limits
  with the OS/container runtime; if the selected platform cannot enforce the profile, refuse it.
- Admit each underlying subject operation through the broker, including operations nested inside an
  approved probe. Reserve prompt plus maximum completion tokens before dispatch, enforce a finite
  generation cap, and count retries. A single fenced block making 1,000 calls stops when its budget is
  exhausted. Raw backend access is confined to approved adapters, never offered to REPL code.
- Drain stdout incrementally into a bounded prefix, discard excess without buffering it, and charge
  all observed bytes against the output quota. Bound each IPC frame and the aggregate response; reject
  oversized messages before allocating their declared payload. Filesystem quotas cover staged blobs;
  the controller validates size and digest before adding any artifact to the immutable store/index.
- On timeout, quota breach, worker death or malformed IPC, stop the outstanding call, log a trusted
  error with the consumed budget, quarantine incomplete blobs, and tear down the affected processes.
  Do not automatically re-execute possibly billed calls or restore arbitrary Python state. Explicit
  resume creates a linked run from the manifest and committed artifacts; verification records the
  failed case as `unreproducible`. Operator cancellation uses the same cleanup path.

**Recursive sub-calls are deferred.** RLM binds a function that lets the model call itself over slices.
A 30-turn transcript fits in a variable, and arbiter's probes handle the slicing. `turn.subcall` is
reserved in `EventKind` so recursion can be added without changing the record format. Add it when a run
shows the need.

**Persistent calculations.** The REPL keeps activation caches and intermediate tensors available across
turns. A stateless JSON interface could do this through `ArtifactRef` ids, so the loop remains
replaceable. Registered probes and model calls cross the typed worker boundary, preserving metering and
event authority when the agent runs arbitrary Python.

**Provider support.** The Claude Agent SDK drives Claude only. This repo's agent role has run on
DeepSeek, Ollama and offline HF judges; Section 7 preserves that support. The SDK's permission modes and
transcript capture could be added later as another driver of the same server, roughly 60 lines.

**The recording proxy.** The controller records an admitted `Call` before execution and correlates the
worker's `Result` afterwards. The worker-side `Subject` proxy does the same for nested operations inside
approved probes. A direct `sub.read(...)` is therefore a call with `probe="subject.read"`. Changing or
replacing a REPL stub cannot manufacture a trusted result or bypass the model-request broker.

Two parts of the current `agent.py` survive: the four-tier structured-output fallback with its
`parse_status` tag (`agent.py:352-430`), added after a real run degenerated into self-talk, and the
word-budgeted incremental reveal (`agent.py:127-154`), which becomes a black-box probe.

### 5.3 The monitor and the audit

AAR runs three Claude monitors in parallel on every method before training: data policy, faithfulness of
the description to the code, and self-containment. They fail closed on infrastructure errors, approval
binds to the exact code hash, and editing forces re-approval. Then a post-hoc pass scans every trajectory
including chains of thought. Arbiter adopts pre-execution review and post-hoc auditing.

**Pre-execution monitor (`monitor.py`, ~90 lines).** Runs on every `turn.code` before the server sees it:

1. A regex deny-list, fail closed: sockets, `subprocess`, `os.system`, any path under `runs/` or
   `fixtures/keys/`, `arbiter.verify`, `save_pretrained`, `torch.save` of anything that is not the store,
   and direct attribute writes on the subject's underlying module. The OS boundary is the real defense;
   the deny-list is the cheap early error with a message the agent can act on.
2. Flagged turns go to an LLM review with one question: does this code read, write, or exfiltrate
   anything outside the subject and the store? Fail closed.
3. The controller records the verdict in a `monitor` event bound to `(request_id, code_hash, manifest_id)`.
   The controller sends `exec` only after approval on that exact tuple. Child messages cannot create or
   reuse approvals. Changed source, dependencies or code require a new manifest/approval as applicable.

**Post-hoc audit (`audit.py`, ~80 lines, `arbiter audit <run>`).** Reads the JSONL, including
`turn.thinking` where available, and flags the three AAR cheating patterns translated to our setting:

| AAR pattern | AAR rate | arbiter translation |
|---|---|---|
| Re-run an unchanged method hoping for noise | 67% of cheats | Same `Call` hash proposed as a finding more than once with different `expect` |
| Training data that copies the benchmark format | 26% | `holdout_kind` claimed broader than any input the agent varied |
| Disguising a banned plan for the reviewer | 21% | Code that passed the monitor after a near-identical rejection, diffed |

The audit writes an `audit` event and never changes a status. A human reads it. AAR's authors note their
agents were "not strongly trying to evade detection" and that this "might not be true for future models".

---

## 6. The plugin contract

A probe is a plain function in a `.py` file. No base class, hand-written package manifest, or entry point
is required. Discovery builds a candidate catalog; researcher-approved source and dependencies are
snapshotted into the execution manifest before a worker imports them. Signatures supply input schemas,
which the broker validates before dispatch.

```python
Lane = Literal["offline", "observe", "parallel", "inline"]

def probe(*, needs: Needs, emits: ResultKind, stimulus: str | None = None,
          lane: Lane = "offline", timeout_ms: int | None = None, name: str | None = None):
    """The registration API. `stimulus` names the kwarg verify may swap for holdout
    inputs; a probe without one can be evidence but never a Finding.check."""
    def deco(fn):
        fn.name, fn.needs, fn.emits = name or fn.__name__, needs, emits
        fn.stimulus, fn.lane, fn.timeout_ms, fn.is_probe = stimulus, lane, timeout_ms, True
        if lane == "inline" and timeout_ms is None:
            raise ValueError(f"{fn.name}: lane='inline' requires timeout_ms")
        if emits == "acts":
            raise ValueError(f"{fn.name}: probes return Results, not bare activations")
        return fn
    return deco
```

**Discovery.** One registry searches `~/.arbiter/probes/*.py`, `./probes/*.py`, `$ARBITER_PROBES`, then
`[project.entry-points."arbiter.probes"]`. Do not keep a separate closed list like harbor's `_AGENT_MAP`.
Reject ambiguous duplicate names.
Discovery locations are not replay locations; worker imports resolve only inside the approved bundle.

Generate the prompt catalog from `inspect.signature` and `__doc__`. Adding a probe touches one file,
the one the researcher owns. Today it touches `_TOOL_NAMES`,
`get_tool_usage_instructions`, the dispatch chain and the cost model.

### Worked example: an SAE feature probe, end to end

Saved as `~/.arbiter/probes/sae.py`, then approved and snapshotted by the researcher. That installs the
probe; the SAE release also needs a checked binding. The REPL gets a callable stub with this signature.

```python
"""Pretrained SAE features over a captured site. Tier W."""
from arbiter import Needs, Site, Result, probe
from arbiter.probes.sae import load_checked_sae

@probe(needs=Needs(caps={"read_acts"}, points={"resid_post"}),
       emits="features", stimulus="text")
def top_features(sub, site: str, text: str, release: str, sae_id: str,
                 k: int = 10, pos: int = -1) -> Result:
    """Top-k firing SAE features at one token position.
    e.g. site="resid_post@12", release="gemma-scope-2b-pt-res",
    sae_id="layer_12/width_16k/average_l0_82" (the l0 suffix is required)."""
    s    = Site.parse(site)
    sae, binding = load_checked_sae(sub, site=s, release=release, sae_id=sae_id)
    acts = sub.read(s, text)[s]
    binding.require_acts(acts.ref)                # exact model, site, preprocessing and shape
    x    = acts.tensor()[0, pos].to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
    f    = sae.encode(x)
    v, i = f.topk(k)
    return Result(kind="features",
                  features=tuple((int(x), float(y)) for x, y in zip(i, v)),
                  source=f"{release}/{sae_id}/{{index}}",
                  artifacts=(acts.ref, binding.weights_ref, binding.config_ref))
```

`load_checked_sae` is an arbiter helper in the shipped SAE probe module, not a SAELens API. It resolves
the release/id through `sub.manifest_id`, checks the binding before capture, and loads frozen local
weights/config on the worker device. It never follows a mutable release alias during replay. It uses
the pinned SAELens version's local loader; release conversion happens once while preparing the bundle.
The binding schema includes model revision, exact `Site` (including head), `coords`,
`preprocessing_id`, input dimension/layout, SAE configuration, and weights/config ArtifactRefs. Upstream
model/hook metadata is a starting point; missing revision/transform details need a researcher-approved
binding plus a golden fixture. The checked binding is itself hashed into the manifest. Its
`require_acts(ref)` repeats identity/layout checks on the actual captured artifact before encoding.

Reject a different layer even if `d_model` matches; reject a different model or revision even if shapes
match. Do not silently authorize base-to-instruct transfer. Preprocessing covers centering, norm
folding, embedding scaling, hook reshaping and SAE input normalization; each transform runs exactly
once. Unknown or conflicting metadata gives a refusal naming the binding that must be supplied or fixed.
See the [SAELens loader metadata](https://github.com/decoderesearch/SAELens/blob/main/sae_lens/loading/pretrained_sae_loaders.py)
for the upstream model/hook fields; arbiter adds the immutable compatibility contract.

Static registration checks capabilities; invocation checks the selected SAE:

```
CompatibilityError: probe 'top_features' cannot use SAE 'layer_12/width_16k/average_l0_82':
  site='resid_post@11' (SAE requires 'resid_post@12')
Fix: pass site='resid_post@12', or select a checked SAE binding for layer 11.
```

Without that check, `sae.encode` can return a perfectly shaped `[16384]` tensor from the wrong layer.
A raw/folded mismatch also refuses when the checked binding requires a different preprocessing profile;
switching to `tl/` is only a fix if that connection's exact profile matches. CPU capture is deliberate:
the example moves the selected activation to the SAE weights' device and dtype before encoding.
GPU memory retained by a probe is charged to that worker's limits and released with its lifetime.

`to_space()` raises by default. Coordinate conversion is allowed only for an architecture with a
checked-in golden fixture (M1).

---

## 7. Model connections

Each named role resolves independently from a provider-prefixed connection. The current code conflates
the subject under study, where hooks attach, with arbiter's judge and investigation models.
Application participants are named separately in `ExecutionManifest.participants`, so a sender/recipient
pair is never confused with the judge or investigation agent. Evaluating a conversation does not require
loading every participant's weights. A deeper investigation binds one selected participant and its model
revision to `subject`; investigating another participant creates the corresponding manifest-bound calls.

```bash
arbiter investigate \
  --role subject=local/google/gemma-2-2b \
  --role agent=openai-compatible/deepseek/deepseek-v3.2 \
  --role judge=openai/gpt-5-mini
```

Specs: `openai/…`, `anthropic/…`, `local/…` (HF weights via nnterp), `tl/…` (TransformerBridge),
`ndif/…` (grey-box), `openai-compatible/…` with a base URL (Ollama, vLLM, OpenRouter, LM Studio).

Structure is pi's three-layer split: inert `Model` data, a wire-protocol adapter keyed by a string, a thin
provider record binding `{baseUrl, auth, catalog, adapter}`. The two wire-protocol adapters are
`openai-completions` and `anthropic-messages`. Only the `subject` role is ever white-box, so tier
requirements apply to that role. `connect()` declares capabilities in `Provides`; see Appendix B for
what hosted providers cannot supply.

This replaces `make_openai_client` (`judge.py:23`), an env-var if-chain returning a bare `AsyncOpenAI`
with the non-OpenAI cases patched in by a duck-typed flag and a base-URL string match. Env vars stay as
defaults.

---

## 8. Evaluation middleware (v1)

The application wraps its existing model client or message-dispatch function with an arbiter adapter.
On each observed exchange, the adapter identifies the session, sender, recipient and parent exchange,
forwards through the existing transport, and submits an immutable capture to the controller. The
controller owns evaluation scheduling and results. The application continues to own model orchestration;
arbiter does not need a second provider-compatible HTTP server.

`arbiter evaluate <transcript> --rubric <rubric.yaml>` evaluates a recorded conversation.
`arbiter watch` drains exchanges captured by the SDK adapter and evaluates them as the application runs.
Both normalize to Exchange records and call the same evaluator with the same
rubric, participant identities, context-window policy and budget. The basic SDK adapter, queue and
offline evaluator are v1 requirements; provider-specific integrations can follow later.

**The evaluation contract.** A frozen rubric declares named criteria, score ranges, allowed flags,
decision thresholds, required evidence, and whether to assess one exchange or a conversation window.
The evaluator may be a deterministic check, a judge-model call, or a registered behavioral probe. The
record includes the exact messages visible to it, the judge/probe version, prompt and sampling settings,
and an Evaluation with scores, flags, rationale and evidence references. Validate outputs against the
rubric before marking an evaluation completed. Evaluation completion describes successful assessment;
it does not claim that a causal interpretation has been verified.

**Default live behavior.** Forward application messages immediately and schedule evaluations alongside
delivery. Capture and queue work have a bounded cost measured by the adapter; judge inference and
investigation are outside the delivery path. Enforce a bounded queue, per-session evaluation budgets and
the controller's global resource limits. Deduplicate source events by authenticated adapter identity,
session and source-event ID, with a payload hash to reject conflicting retries. Never resend an
application message as a side effect of retrying an evaluation.

Sample whole sessions when sampling is configured. Record selected context, sampling policy and
coverage so a partial assessment cannot appear to cover an entire conversation. Queue overload emits a
visible drop record; once an evaluation is assigned an ID, dropping it produces `status="dropped"`.
Evaluator timeout/failure produces `status="error"`. Neither is converted to a low-risk or passing score.
Applications consume structured result callbacks or read the run record; a report summarizes per-model
and per-session scores together with evaluated, dropped and failed counts.

**Escalation to investigation.** A configured rubric flag can enqueue a deeper investigation against
the selected participant. The transcript and evaluator rationale enter as untrusted data in the REPL,
with only metadata and bounded observations sent to the agent model. Its probes may produce proposed
findings. `arbiter verify` independently checks those findings under Section 4.4; linking a finding to an
Evaluation does not promote its status. Investigative patches run on isolated subject copies.

**Additional adapters.** LiteLLM callback integration, OTLP ingestion and optional bounded inline
evaluation are later extensions of this contract. They must preserve correlation, event authority,
coverage reporting and the default delivery policy. The earlier LiteLLM hook inventory is implementation
research, not a requirement to put a specific proxy between every pair of models.

**Hosted-model access.** Hosted traffic is black-box only. OpenAI encrypts reasoning as
`encrypted_content`; Anthropic's `signature` is an encrypted copy of the thinking and raw CoT is never
returned. White-box monitoring needs self-hosted traffic and a collector inside the serving engine. vLLM
ships hidden-state extraction as of v0.18.0 (a dummy speculative-decoding method plus a KV connector,
returning a `.safetensors` path), which fits `ArtifactRef`. v1 ships an adapter for that file, not an
in-engine collector. RFC #36998 is open with no implementation.

**Memory promotion.** Live exchanges, evaluation rationales and `provenance == "tap"` findings
are untrusted data. They do not become prompt instructions or write directly to the recipe library or
ledger. Promotion happens offline, gated by a re-run and the recipe review process.

---

## 9. Self-learning

**Tier 1, cross-investigation aggregation (build now, ~110 LOC).** Every run writes
`findings/<model_id>/<run_id>.json`. A new investigation reads the last N. `--aggregate 10` runs N
separate investigations and feeds all N reports to a fresh agent. Anthropic's auditing agents went
from 13% to 42% on the auditing game with this aggregation method.

Bowkis et al. (2026) caution that N runs of the same agent model are correlated evidence.
Aggregation raises hypothesis diversity; it does not give independent trials. Do not report "8 of 10
runs agree" as if it were a p-value.

**Tier 2, a recipe library (~100 LOC).** `recipes/<name>/SKILL.md` with name/description frontmatter plus
an optional `probe.py`. A run that produces a verified finding may propose a new recipe as a git diff
for human review. Revision 1 required Tier 1 to show a measured improvement before building Tier 2.
AAR's environment ablation supports building the library earlier:

| Removed from AAR | Headroom lost (sycophancy, single runs) |
|---|---|
| Literature review | ~9 points (29.4% → 20.7%) |
| Finding forum | ~6 points (23.3% → 17.1%) |
| Internet access | none detectable |

Removing literature review cost more headroom than removing the forum. These were single mitigation
runs, so the result does not establish a benefit for auditing. Build Tier 2 as soon as Tier 1 exists,
and measure both on the fixtures before relying on them.

**Tier 3, a per-model findings ledger (gated).** Append/amend deltas with run id, model revision and a
re-verification status. Re-verify on a new revision.

**Tier 4, JSONL trajectory logging.** Already covered by the recorder.

**Out of scope:** vector-DB memory, Reflexion loops without a verifier, RL fine-tuning on trajectories,
a bespoke agent framework, and self-modifying prompts. Changing prompts between runs invalidates
cross-run comparability. Memory edits are typed, diffable, revertible, and confined to a store whose
version is recorded in every run.

The M3 fixtures test whether investigation N, using a populated ledger and recipe library, beats
investigation 1 on planted-quirk targets.

---

## 10. Module map and LOC budget

| Module | LOC | Role |
|---|---|---|
| `arbiter/__init__.py` | 30 | Lazy `__getattr__` so `import arbiter` never imports torch |
| `arbiter/contracts.py` | 340 | Exchange/Evaluation records, investigation types, selectors, validation |
| `arbiter/store.py` | 180 | Staging/commit, digest and provenance validation, bf16-safe storage, retention |
| `arbiter/record.py` | 160 | The only JSONL writer; peer authority, schema validation, proposal hashes |
| `arbiter/manifest.py` | 180 | Immutable source/model/artifact bundles, resolved settings, cache identity |
| `arbiter/registry.py` | 130 | Candidate discovery, approval/snapshot, static and invocation compatibility |
| `arbiter/connect/__init__.py` | 80 | `connect(spec)`, role resolution |
| `arbiter/connect/api.py` | 190 | OpenAI + Anthropic + compatible, blocking, retries, capability declarations |
| `arbiter/connect/local.py` | 260 | nnterp `StandardizedTransformer`; harvests the debugged hardware layer |
| `arbiter/connect/tl.py` | 120 | TransformerBridge behind the same `Site` vocabulary |
| `arbiter/connect/remote.py` | 90 | NDIF / grey-box |
| `arbiter/connect/proxy.py` | 80 | RecordingProxy |
| `arbiter/server.py` | 180 | Separate REPL and subject-worker entrypoints; typed IPC, bounded streams |
| `arbiter/broker.py` | 180 | Trusted request admission, token reservations, process/resource supervision |
| `arbiter/middleware.py` | 150 | SDK handoff adapter, bounded queue, correlation and coverage |
| `arbiter/evaluate.py` | 130 | Shared live/offline rubric evaluator and structured results |
| `arbiter/loop.py` | 100 | Metadata prompt, fenced-code turns, FINAL/FINAL_VAR, budget |
| `arbiter/prompt.py` | 90 | Catalog from signatures + docstrings; immutable versioned base prompt |
| `arbiter/monitor.py` | 90 | Deny-list + LLM review, fail closed, code-hash binding |
| `arbiter/verify.py` | 220 | Frozen replay, paired baseline/treatment trials, alignment, Wilson interval |
| `arbiter/audit.py` | 80 | Post-hoc trajectory flags |
| `arbiter/fixtures.py` | 100 | Transcript loader and causal-fixture loader; private keys |
| `arbiter/memory.py` | 110 | Tier-1 aggregation, Tier-2 recipe discovery |
| `arbiter/config.py` | 60 | Strict, layered replace, `--dump-config`, package-relative |
| `arbiter/cli.py` | 190 | evaluate / watch / capture / probe / investigate / verify / audit / plugins / dump-config |
| **Core** | **3,520** | |
| `arbiter/probes/{sae,lens,patching,vector,blackbox}.py` | 380 | Shipped probes, each a worked example |
| **Total v1** | **~3,900** | Includes basic evaluation middleware; additional adapters are later. |

These are planning estimates for production Python; tests, fixtures and environment definitions are
counted separately. The estimates include the manifest, trusted broker and evaluation middleware. CI
reports size against a 4,000-line target; exceeding it triggers a scope review, never removal of validation or an
isolation gate. Additional adapters and memory slip first; basic exchange capture and evaluation remain
part of v1.

v0 starts with two files: `contracts.py` and one `arbiter.py` holding everything else. Split modules
when their responsibilities need separate maintenance, using the table as the target layout.

**Config uses layered replacement.** An override restates the fields it keeps, replacing the deep-merge
design documented in `CLAUDE.md`. `arbiter dump-config` renders the composed result through the real boot
path. Also fix `config.py:9`, which resolves outside the package.

---

## 11. Dependencies

```toml
dependencies = ["httpx", "pyyaml", "safetensors"]      # the whole base install; no torch

[project.optional-dependencies]
local = ["torch", "transformers", "nnsight>=0.7", "nnterp>=1.3"]
tl    = ["transformer-lens>=3.8"]                      # transformers>=5.9; conflicts with ct
sae   = ["sae-lens>=6"]
ct    = ["circuit-tracer @ git+https://github.com/decoderesearch/circuit-tracer"]
        # NOT the PyPI package of that name, which is an unofficial fork with looser pins
```

`tl` and `ct` are mutually exclusive by upstream pins. Probes with incompatible pins run
out-of-process and communicate through `ArtifactRef`.

A test asserts `"torch" not in sys.modules` after `import arbiter`.

The ranges above describe install compatibility, not replay pins. Resolve each worker environment to a
committed lock with exact versions and git commit IDs before use; record its digest and the source
bundles in the execution manifest. A bare git branch or `main` is never an execution identity.

Dropped: the pinned flash-attn wheel URLs (a cp313 linux_x86_64 URL in a base manifest is a portability
trap), matplotlib, datasets, peft, outlines, openai, python-dotenv.

---

## 12. Deletions and retained fixtures

About 300 of the package's 2,973 lines survive. Another 1,006 lines of root-level sweep scripts go.

| Deleted | LOC | Why |
|---|---|---|
| `arbiter/agent.py` | 962 | `run_agent_loop` is one 454-line function (`:509` to EOF) with `if tool_name ==` branches at `:666/:732/:768/:856/:914`, each with inlined budget arithmetic. **Salvage** the structured-output fallback (`:352-430`) and the incremental reveal (`:127-154`). |
| `arbiter/tools/` | 261 | `_TOOL_NAMES` serves only DESCRIPTION strings while dispatch happens elsewhere; three tools sync, two async. `inspect_cot` and `inspect_system_prompt` are dict lookups over pre-recorded text. Module-globals `_model_cache` and `_incident_log` break concurrency. |
| `arbiter/experiments.py` + `run_experiment.py` + `analyze_experiments.py` + `generate_conversations.py` | 1,355 | Sweep orchestration is a job runner. `GROUND_TRUTH = "Maria"` is hardcoded at `:27`. |
| `arbiter/judge.py` | ~450 of 581 | The env-var if-chain, the duck-typed offline flag, the base-URL string match, all asyncio. **Retain about 150 lines from `OfflineInferenceJudge`'s hardware layer (`:161-436`)**: dtype resolution, left-padding, 4-bit, static-KV-cache fallback, all debugged against real hardware. |
| `arbiter/core.py` | 196 | Absorbed into `connect/local.py`. **Salvage the LoRA auto-detect (`:27-35`).** |
| `arbiter/plot.py`, `summary.py` | 162 | Replace legacy file-format-specific reporting with Evaluation exports and coverage summaries. |
| CLI `run`/`judge-dataset`/`plot`/`summary`/`experiments` | ~270 | Five of seven verbs. |
| `examples/ag2_misalignment_demo.py` + the `ag2` extra | 429 | Conversation generation is outside arbiter's scope. M3 imports Petri/Bloom transcripts (Section 13). |
| 79 `print()` calls | n/a | Replace direct printing so the library can run in a server process. |
| `results.json` as the canonical format | n/a | Survives as one ~40-line exporter. |

Keep the sweep outputs that revision 1 would have deleted. `results/` holds transcript
detection fixtures: a misaligned agent named Maria, colluding pairs, aware and unaware system-prompt
variants, and repeated detection measurements in `analysis_stats.md`. Move these to
`fixtures/transcripts/`, splitting public stimuli from private answer keys, hidden facts and scoring
metadata under `fixtures/keys/`. Do not leave readable duplicate keys in source JSON or directory labels.
These fixtures measure detection, not a known causal tensor mechanism. M3 adds the latter separately.
Keep the outputs; do not delete any generating code until M3 has loaded and validated its fixtures.

Also: `CLAUDE.md:26` is already corrected in the working tree; commit it.

---

## 13. Milestones

Each milestone ships something runnable. M1 gates backend support; M3 must pass before the agent ships.

- **M0: Contracts, manifest and store, no agent.** Capture `resid_post@12` from gemma-2-2b to an
  `ArtifactRef`; a hosted subject must refuse and name the missing capability. Freeze source, model,
  tokenization and input artifacts; replay the bundle after changing the discovery directory. Changing
  a probe, backend preprocessing, model revision or input artifact must change execution/cache identity.
- **M0.5: The pipe and authority boundary.** Launch the REPL and subject worker as distinct restricted
  processes, with the controller broker between them. Before any loop exists, prove the REPL cannot access
  `runs/`, keys, GPU devices, worker memory/socket, or provider credentials. Forged verification/monitor
  payloads, reserved proposal fields, replayed request IDs and oversized IPC frames must be rejected.
  An infinite loop, model-call loop, stdout flood and artifact/disk flood must hit externally enforced
  limits, terminate cleanly, and leave no committed partial artifact or outstanding uncharged request.
- **M1: Substrate, with a golden-fixture gate.** `connect/local.py` and `connect/tl.py` behind one `Site`
  vocabulary. Freeze this gate before running: in fp32, `resid_post@N` from nnterp must match a
  TransformerLens cache loaded with `fold_ln=False, center_writing_weights=False, center_unembed=False`
  (or TransformerBridge without compatibility mode) within 1e-4 absolute, on gpt2 and on gemma-2-2b.
  A second fixture turns compatibility mode on and asserts the same site now differs from raw HF and
  matches `HookedTransformer` defaults within 1e-4, testing that `coords` describes the captured data.
  Revision 1 said bf16 and default TL processing, which fails by construction: bf16's relative spacing
  is ~8e-3, and TL's centering changes the residual stream. If the fp32 Gemma fixture fails, believe the fixture and
  reverse the substrate decision. Do not loosen the tolerance.
- **M2: Registry and the first probe.** Candidate discovery/approval, `Needs.unmet`, catalog from
  signatures, and the checked SAE probe from Section 6. Gate on valid encoding on CPU and GPU, including
  CPU-captured activations sent to a GPU SAE. Wrong model/revision, wrong same-width layer, incompatible
  preprocessing, unknown binding and wrong dimensions must refuse before encoding. The binding fixture
  checks the selected SAE's features/reconstruction against its reference capture, not just HF/TL parity.
- **M3: Fixtures, findings, verify. Before any agent.** Import transcript detection fixtures and their
  historical scores, then add `fixtures/causal/tiny-trigger/`: a deterministic CPU `nn.Module` with fixed
  weights, a fixed tokenizer and a known trigger dimension at `resid_post@0` on the final prompt token.
  Its source and rebuild recipe are public and pinned; verification cases/keys are evaluator-only. It
  requires no training, model download or LLM judge. This tests the controller; M1 covers real backends.
  Use the precise behavioral definition and acceptance cases below. Import one Petri/Bloom transcript
  as an input-format compatibility fixture; it is a format holdout only when paired with the same
  content/behavior in another wrapper. An unrelated transcript alone does not establish format transfer.
  Add a versioned transcript-evaluation rubric with known output/flag expectations and coverage checks.
- **M3.5: Basic evaluation middleware.** Connect two model clients through the SDK adapter,
  capture correlated exchanges, forward them through the existing transport, and emit rubric-based
  Evaluation records. Replay those exchanges through `arbiter evaluate` using the same context policy.
  Gate on duplicate capture handling, accurate evaluated/dropped/error counts, and no message redelivery
  when the judge fails or an evaluation retries. A deterministic evaluator makes live/offline parity a
  CI assertion; model judges are tested under their recorded sampling settings. Then wire one tier-B
  investigation through REPL, worker, metered broker and monitor, ending in a proposed finding.
- **M4: Investigation agent.** Hardened loop, LLM monitor stage, `arbiter audit`, `turn.thinking` logged
  where providers return it. Report detection rate on the transcript fixtures and verified causal
  findings on the causal fixtures separately. Historical `analysis_stats.md` supports detection
  comparisons only; match tool access and model/token budgets or label the comparison as unmatched.
- **M5: Portability demo.** The same unmodified probe file against `local/gemma-2-2b`, `tl/gemma-2-2b`,
  and `ndif/Llama-3.1-70B`, exercising the plugin contract across backends and access tiers.
- **M6: Additional middleware adapters.** Provider-specific hooks and optional inline evaluation.
  These can slip; the basic SDK adapter, offline evaluation and watch loop ship in M3.5.
- **M7: Memory.** Tiers 1 and 2, with the falsifiable claim from Section 9 measured on the M3 fixtures.

**M3 causal fixture definition.** The tiny model's final-token residual has a designated first
dimension equal to 1 when the prompt contains the trigger token and 0 otherwise. Other dimensions and
the other exposed site do not affect its two output logits. The approved metric is
`logit_gap = logit_A - logit_B = 2 * trigger_dimension - 1`. The registered intervention sets the final
`resid_post@0` vector to a stored zero vector, with `PositionSelector(kind="last")`. Positive prompts
therefore have baseline +1, treatment -1 and delta -2; negative control prompts have -1, -1 and delta 0.
Keep each case's baseline/treatment input identical, and vary prefix length and trigger position across
cases. The real-model SAE gate remains separate from this synthetic model.

**M3 acceptance cases (all required):**

| Change / input | Required result |
|---|---|
| Eight distinct positive/control pairs, `expect=-2`, `control_expect=0`, `n=k=8` | Verified; all 32 arm results and resolved positions recorded |
| A no-op treatment with the same nonzero expected effect | Refuted; matching absolute outputs cannot substitute for a delta |
| Remove the control, change only its treatment, or label a patched call observational | Unfalsifiable |
| Edit k from 8 to 1 or edit n after proposal | Old proposal hash invalid; no verification under the old identity |
| `n=0`, `k=0`, `k>n`, non-integer thresholds or non-finite numeric fields | Reject at proposal ingestion |
| Corrupt zero-vector bytes without changing its content address | Unreproducible; integrity failure before execution |
| Commit a new vector with trigger dimension 1; submit a new proposal with `expect=-2` | Refuted; valid new artifact identity but the claimed effect is absent |
| Paraphrase to different token lengths / move trigger location, retaining `last` selector | Correctly realigned final-token intervention in every case |
| Fixed token indices, ambiguous spans, or donor tensors tied to the old sequence | Refuse under the selector/alignment rules; never silently retarget |
| Timeout, invalid alignment or missing holdout case | Unreproducible with the failure recorded; n and k unchanged |
| Change plugin source or processing defaults after the run | Frozen bundle replays; missing bundle refuses, never falls back to discovery |

Proposal/IPC rejection tests and CPU causal tests run in ordinary CI. GPU SAE tests and real-model M1
fixtures run in the supported accelerator environment before those backends ship. Keep these as
behavioral gates rather than tests that only reconstruct the dataclass definitions.

---

## 14. Limits

State these in the README.

- In Anthropic's open-ended auditing game, the interpretability agent found the true root cause about
  13% of the time, 17% when seeded, and 42% with parallel aggregation. Build for cheap repetition and
  human-checkable evidence, never for a single verdict.
- Scaffolded black-box tools beat interpretability tools on average in AuditBench, above 50% against a
  ~37% baseline. Interpretability's wins are bimodal: Activation Oracles hit 88% vs 0% on one behavior and
  beat the default on 8 of 14 in the adversarially trained subset; SAEs and logit lens are mixed.
  Section 12 keeps the black-box fixtures, and M3 adds Petri/Bloom transcripts.
- AAR's audit found cheating in 2.4% of trajectories, from agents not trying to evade. Treat every
  `proposed` finding as adversarial input to `verify`.
- Runs using the same weights and training data have correlated errors; they are not independent trials.
- A completed evaluation reports an assessment under its rubric. A verified observation reproduces on
  fresh inputs; a verified intervention also reproduces a nonzero paired effect while its control
  remains unchanged. Reproducibility still does
  not establish that the interpretation is right.
- SAEs and transcoders exist for a handful of models. For most subjects, tier W means hooks, probes
  and patching.
- Backend maintenance is ongoing. TransformerLens shipped two releases in ten days.
  Pin deliberately, wrap thinly, budget for it.

---

## 15. Open questions

1. **Which application should integrate first?** The core SDK adapter supports model handoffs and hosted
   text evaluation. Choose the first application's transport and rubric; local internals add deeper probes.
2. **Should the emergent-misalignment tooling remain a separate product?** Section 12 keeps its fixtures
   and deletes its scripts. An alternative is to keep `run`/`judge`/`plot` in a separate package.
3. **Which second architecture should M5 use?** Gemma-2-2b is AAR's deception target and has Gemma Scope
   coverage. AAR's other targets are Qwen3.5-2B, Llama-3.2-3B, Phi-4-mini and Olmo-3-7B.
4. **Which agent model?** The loop is provider-agnostic, but the monitor and audit want thinking blocks,
   and only some providers return any. If the agent is usually DeepSeek or a local model, `turn.thinking`
   will usually be empty, and the audit leans on code alone.

---

## Appendix A: Sources

| Source | Taken |
|---|---|
| earendil-works/pi (MIT) | Three-layer Model/API/Provider split; tool `content` vs `details`; ship-nothing-in-core |
| PrimeIntellect-ai/prime-agent (MIT) | Single-`python()`-tool design; skill-as-directory discovery; immutable base prompt |
| harbor-framework/harbor (Apache-2.0) | Entry-point plugin group; `uv run` boundary for incompatible dependencies; declared and checked capabilities |
| laude-institute/headlong (Apache-2.0) | Append-only JSONL as bus and audit trail; blob spill with a recoverable ref |
| deepseek-ai/deepseek-harness (MIT) | "Model-visible implies logged"; record-once-replay-forever |
| walkinglabs/learn-harness-engineering (MIT) | Generator/evaluator separation; fail-closed tool registry |
| nexu-io/harness-engineering-guide (MIT) | Small controller with task logic in skills; triage before full investigation |
| YuehHanChen/automated_alignment_researcher (Anthropic, 2026-08-28) | OS-user isolation of held-out data; three-monitor pre-execution code review bound to a code hash; results-free pre-registration; capability gate; scenario/domain/format holdouts; post-hoc trajectory audit; the environment ablation |
| Petri (Meridian Labs, v3.0) and Bloom (safety-research/bloom) | Tier-B transcript sources; Inspect-format JSON as a `--transcript` input |
| Bowkis, Buhl, Pfau, Irving, "Automated alignment is harder than you think" (arXiv 2605.06390) | Correlated-runs caveat in Sections 9 and 14 |

Also: Inspect AI for `model_roles` and both white-box provider examples; pyvene for config-as-data
interventions; Neuronpedia for the addressing triple; the RLM paper for bounded observations and the
symbolic handle.

## Appendix B: Fact-check corrections

Corrections to earlier revisions. Items 1–11 are from revision 1, re-verified 2026-09-01; items 12–17
were added in revision 2.

1. RLM outperforms CodeAct by a median of 130% on GPT-5. The "JSON tool-calling standards" quote appears
   nowhere in the paper. The `rlms` package (0.1.3, 2026-06-26) exists but is not an arbiter dependency;
   its default sandbox executes model-written Python in-process.
2. TransformerBridge preserves raw HF weights by default. Folded space requires
   `enable_compatibility_mode()`, which calls `process_weights` in place; there is no undo method.
3. TransformerLens 3.8.x requires `transformers>=5.9.0`. circuit-tracer pins `nnsight==0.6.1` and
   `transformers<=4.57.3`.
4. SAELens 6.x (6.50.0 as of 2026-08-30): `SAE.from_pretrained(release, sae_id, device=...)` returns the
   SAE alone; `from_pretrained_with_cfg_and_sparsity(...)` is the three-way form. Repo is
   `decoderesearch/SAELens`.
5. circuit-tracer is hosted under Decode Research and supports no Claude models. Two of its four
   credited authors are Anthropic staff; "maintained by Decode Research, not Anthropic" is an inference
   from the org move, not a stated fact.
6. Anthropic never returns raw chain of thought. `thinking.display` has three values: `summarized`
   (default through 4.6), `omitted` (default on 4.7/4.8 and the 5 generation), and beta `updates`.
   `signature` is an encrypted copy of the full thinking. There is no `logprobs` parameter.
7. captum is active (0.9.0, 2026-04-17, not May). pyvene last released 2025-05; repeng 2024-12;
   tuned-lens 2023; ecco 2022. baukit is not on PyPI.
8. vLLM hidden-state extraction shipped in v0.18.0 (PR #33736), prompt tokens only, chunked prefill
   disabled. RFC #36998 is open with no implementation.
9. "Hook-based observability costs 1.4% prefill / 5% decode" is invented.
10. LiteLLM overhead: avg 3.32 ms, median 2 ms, P95 8 ms on a four-instance deployment against a mock
    endpoint, Postgres, no Redis.
11. "Interpretability tools lift win rates 10–20% from near 0%" is not what AuditBench reports.
12. The RLM "13% of turns confused FINAL vs FINAL_VAR" was a compression. The paper's Appendix A says
    16% of turns misused `FINAL` and 13% misused `FINAL_VAR`, measured on Qwen3-Coder teacher trajectories
    during data cleaning, not on GPT-5 evaluation runs.
13. `sae_id="layer_12/width_16k"` is not a valid Gemma Scope id. The release has exactly five
    variants under that path, `average_l0_{22,41,82,176,445}`, or use `gemma-scope-2b-pt-res-canonical`
    with `/canonical`.
14. The PyPI package `circuit-tracer` (0.5.0) is an unofficial fork with `nnsight>=0.6.0`. Pin the git
    source.
15. The revision-1 M1 gate could not pass. 1e-4 absolute in bf16 is below the format's resolution, and
    `HookedTransformer` defaults center writing weights, which changes `resid_post`. See Section 13.
16. "Lane 0 emits `x-arbiter-overhead-ms`" was impossible. `async_log_success_event` runs after the
    response is sent. Use `async_post_call_response_headers_hook`.
17. Revision 1's rule 5 was an honor system. AAR's `ISOLATION.md` says of their own earlier design
    that leaving a research-readable copy of the keys "would have let the AAR read the answer keys
    directly and defeat the lock." Section 5.1 is the fix.

## Appendix C: What AAR does, and what arbiter takes

| AAR mechanism | Arbiter equivalent | Where |
|---|---|---|
| Held-out data under a separate OS user, mode 700; agent polls for scores | Distinct REPL, subject worker and evaluator identities; authenticated event authority and metered requests | 5.1 |
| Three parallel code monitors, fail closed, approval bound to code hash | Deny-list + LLM review per `turn.code`; controller binds approval to request, code and manifest | 5.3 |
| Post-hoc trajectory monitor over chains of thought | `arbiter audit`; `turn.thinking` logged | 5.3 |
| Results-free mini-paper frozen to an immutable id before results | Recorder freezes full proposal, n/k policy and execution manifest before evaluation | 4.4 |
| Capability gate: 95% CI vs untrained baseline | Paired effect/control deltas against unpatched baselines; Wilson interval printed, not a CI gate | 4.4 |
| Hill-climbing benchmarks + one held-out of kind scenario/domain/format | `holdout_kind`, `verified_on` | 4.4 |
| Submit a model, never predictions | Submit manifest-bound calls; verifier computes deltas from fresh worker measurements | 4.4 |
| Shared forum and leaderboard across five parallel agents | Tier-1 aggregation over `findings/` | 9 |
| Literature review phase | Tier-2 recipes, no longer gated on Tier 1 winning | 9 |
| Petri as the open-ended held-out audit | Petri/Bloom transcript import; format-transfer claims require paired same-content wrappers | 13, M3 |
| Activation steering to build a better teacher | `ArtifactRef(kind="vector")` as the diagnose→mitigate handoff | 0 |
| Not adopted: fresh session per iteration; training loop; forum server; Claude Agent SDK | Persistent REPL; no training; file-based aggregation; provider-agnostic loop | 5.2 |
