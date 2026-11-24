# Real-adapter build notes (from source reads, 2026-11-24)

## artifacts.py (532 lines) — key APIs
- `file_digest(path) -> "sha256:..."`; `_tensor_digest(tensors) -> digest_model("checkpoint-tensors", payloads)` (sorted names, dtype/shape/data_sha256 per tensor).
- `load_tensors(path, max_bytes=1GiB)`: float32/float64 only, finite check. `_check_safetensors_file`: header ≤1MiB.
- `validate_graph_tensors(graph, tensors)`: tensor set must EXACTLY equal graph writer tensor_names; shape+dtype match; ndim==2.
- `ArtifactStore(root)`: `.path(rel)` (no escape), `._unique(stem,suffix)` (stem `^[a-z0-9][a-z0-9_-]{0,63}$`, refuses overwrite), `.write_graph(graph, artifact_id) -> f"{id}.graph.json"`, `.load_graph(rel)`, `.write_direction(dir, id)`, `.load_direction(rel)` (dispatch on schema_version), `.write_bank(bank,id)`.
- `.save_checkpoint(tensors, graph, artifact_id) -> ArtifactReference(role="baseline")`: validates, saves safetensors w/ metadata {dwmip_version:"0", checkpoint_id, graph_digest, artifact_id, dtype_contract:"float32-or-float64"}, readback digest check.
- `.reference(rel, role, graph, tensors=None) -> ArtifactReference` (artifact_id = filename minus .safetensors).
- `.derive_candidate(candidate, graph, tensors, writer_id, artifact_id) -> BoundDirection`: builds in-memory baseline ref (artifact_id="in-memory-baseline", path="in-memory", file_digest=digest_bytes("in-memory", tensor_digest.encode())); requires candidate.baseline_digest == tensor_digest AND graph_digest match; coordinate site unique + space match; candidate.space_id == writer.output_space_id; `candidate.bind(writer_id, tensor_name, dtype, shape)`; persists bound direction.
- `.compile(graph, baseline, intervention, artifact_id, direction_path=None) -> CompiledArtifact`:
  - `_validate_execution(graph)` FIRST (see below).
  - direction_path: persisted BoundDirection must == intervention.direction.
  - observed baseline reference must == passed baseline; role=="baseline"; artifact_id matches path stem.
  - direction.baseline_digest == baseline.tensor_digest; direction.graph_digest == graph.digest().
  - writer = _find_writer(graph, direction.writer_id); _validate_writer_binding (tensor_name,dtype,shape,output_space_id).
  - MATH: unit=direction.array(); projection = unit @ original.astype(f64); delta = alpha*np.outer(unit,projection); edited = orig.f64 - delta; result = edited.astype(orig.dtype).
  - candidate metadata: {dwmip_version, checkpoint_id, graph_digest, artifact_id, baseline_tensor_digest, baseline_file_digest, direction_digest, intervention_digest=digest_model("intervention",intervention), algorithm_id}.
  - structural: observed per-tensor digests == expected; non-target unchanged; max deviation ≤ tol = eps(orig.dtype)*max(1,max|orig|)*4; distinct paths. Fail → unlink candidate + raise.
  - manifest at f"{artifact_id}.manifest.json" (refuses overwrite); metadata re-check; `_structural_check(...)`; CompiledArtifact(adapter_id=graph.adapter_id, baseline, candidate, intervention, direction_digest, structural_check).
- `.verify_compiled(rel)`: parse manifest; graph from f"{baseline.artifact_id}.graph.json"; adapter/graph digest match; _validate_execution; both refs re-verified; tensors reloaded; metadata exact; direction binding+digests; independent recompute expected = orig.f64 − alpha*outer(v, v@orig.f64); _structural_check recomputed must == manifest.structural_check.
- `_validate_execution(graph)`: **adapter_id must == "synthetic-cpu-v0"** ← THE GATE TO EXTEND. topology=="dense_linear", execution_supported; no recurrent_step/loop_iteration/shared_group; writers operations==("rank_one_output",); one writer per site, one site per writer coordinate.
- `_one_tensor_digest`, `_per_tensor_digests`, `tensor_metadata(path)` (safe_open metadata only).

## interventions.py
- BoundDirection: schema "dwmip.bound-direction.v0"; unit_vector (writer output dim, finite, norm≈1 within unit_norm_tolerance); coordinate: SiteCoordinate; writer_id/tensor_name/dtype/shape/space_id; baseline_digest, graph_digest; capture bank fields (candidate.bind carries them); `.array()`, `.identity_payload()` (excludes unit_vector), `.direction_digest() = digest_model("bound-direction", self)`.
- InterventionSpec: schema "dwmip.intervention.v0"; operation Literal "weight_rank_one"; algorithm_id Literal UPDATE_ALGORITHM_ID; direction: BoundDirection; alpha ∈[0,1]; coordinate property = direction.coordinate.
- ArtifactReference: artifact_id, role baseline|candidate, path, codec "safetensors", file_digest, tensor_digest, checkpoint_id, graph_digest.
- StructuralCheck: baseline, candidate, operation, algorithm_id, direction_digest, alpha, expected_tensor_digests, candidate_tensor_digests, target_digest_before/after, maximum_deviation, tolerance, passed.
- CompiledArtifact: schema "dwmip.artifact.v0"; adapter_id; baseline; candidate; intervention; direction_digest; structural_check; codec_capabilities ("read","write","structural_audit"); efficacy/safety "unmeasured"; release_eligibility "not_evaluated". Validators: direction_digest match; distinct artifact ids; nonzero alpha ⇒ different file digests.

## banks.py
- BankRole = capture|calibration|evaluation|release_redline|mock_release_redline. PUBLIC_LICENSES={CC0-1.0,CC-BY-4.0}.
- LicenseTerms: spdx_id, copyright, source, license_text, redistribution_allowed(bool|None), transformation_lineage, generator_terms.
- BankRecord: record_id, prompt(≤32k), group benign|hostile|control, domain, pair_id(optional).
- BankManifest: schema "dwmip.bank.v0"; bank_id, version, role, visibility public|private, records(1..100k), license, split_policy, notes. capture role: ≥4 balanced benign/hostile pairs, every record pair_id, each pair one benign+one hostile. release_redline must be private. private cannot be redistributable. `.digest()=digest_model("bank-manifest",self)`, `.reference() -> BankReference(bank_id,version,role,visibility,manifest_digest,records_digest=sha256(canonical_bytes([r.model_dump])),count)`.
- require_publicly_bundlable: public + redistribution True + approved SPDX + role != release_redline.

## synthetic.py
- reference_graph(dtype): GraphSpec(adapter_id="synthetic-cpu-v0", checkpoint_id="synthetic-reference-v0", topology="dense_linear", execution_supported=True, spaces=(output SpaceSpec(space_id,kind,dimension,basis,normalization),...), sites=(SiteSpec(coordinate=SiteCoordinate(site_id,logical_layer), space_id, capture_timing),...), writers=(WriterSpec(writer_id,tensor_name,coordinate,output_space_id,input_space_id,dtype,shape,operations=("rank_one_output",)),...)).
- synthetic_capture_bank(): BankManifest(bank_id,version="v0",role="capture",visibility="public",records,license=LicenseTerms(...),split_policy,notes).
- capabilities() -> AdapterCapabilities(adapter_id, executable_topologies, supported_operations, supported_codecs).
- SyntheticObservation contract: schema "dwmip.synthetic-observation.v0", observation_id, graph_digest, baseline/candidate_tensor_digest, coordinate, writer_id, input_vector, baseline_output, candidate_output, behavior_class="diagnostic_only".

## Design decision for real adapter
- New adapter id "hf-dense-linear-v0" registered in _validate_execution alongside synthetic-cpu-v0 (same dense_linear rules: no loop/recurrent/shared, one writer per site). Graph declares ONLY the writers being edited?? NO — validate_graph_tensors requires tensor set == writer set, so the real graph must declare ALL tensors we store. For real HF checkpoints: declare every 2-D weight as WriterSpec? That forces operations=("rank_one_output",) on all and one site per writer — heavy but doable for small dense models: declare all Linear weights as writers+sites+spaces. Non-writer tensors (norms, embeddings 2D?) — embeddings are 2D too; biases 1D would fail ndim==2. OPTION: store only 2-D tensors in the DWMIP checkpoint (the "editable surface"), keep 1-D/other tensors in the HF snapshot untouched; bake writes back only the edited tensor into a copy of the HF snapshot. Graph = editable surface (all 2D float tensors), each with site+space. That satisfies exact-set validation.
- dtype: HF float32 fine; bfloat16 NOT supported by load_tensors → convert to float32 at intake (record conversion in provenance) or refuse. v0: convert bf16/f16→f32 explicitly, record original dtype in graph notes/provenance; bake back to original dtype on export.
- S2 capture: torch hooks on module outputs (o_proj/down_proj etc.) mean-difference hostile vs benign over pack capture corpus → DirectionCandidate per coordinate (correlational label).
- S3 bake: pick coordinate+alpha (operator param), bind via store.derive_candidate, InterventionSpec, store.compile → manifest; export edited tensor back into HF-format safetensors copy (same dtype as original).
- Evidence: emit per-stage JSON in run dir + DWMIP manifest; unmet stages (S1 baseline bench, S4 quant, S5 serve, S6/S7, SR) recorded NOT_RUN with reason via control store skip/record paths.