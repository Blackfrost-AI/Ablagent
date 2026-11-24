# Navigating architectures without guessing

## Adapter qualification is mandatory

A graph representation is not a forward-pass implementation. A real adapter must demonstrate how its loader, capture mechanism, writer mapping, codec, and serving path correspond to the same checkpoint and runtime.

Every adapter declares:

- `GraphSpec`: logical sites, topology, sites and writers, shared parameter groups.
- `SpaceSpec`: basis, dimensions, normalization, and inner product.
- `WriterSpec`: tensor identity, axes, packing, input/output spaces, supported operations.
- `SiteCoordinate`: layer and optional route, expert, loop iteration, recurrent step/state.
- Capture timing, pre/post normalization semantics, in-place operations, and expected coverage.
- Exact codec and serving identities, including fused-kernel and quantization behavior.

Missing semantics return `UNSUPPORTED`. Do not substitute a tensor-name regex, shape heuristic, or a sibling model's recipe.

## Dense residual networks

Separate pre-mixer input, attention output, post-attention residual, pre-MLP residual, MLP output, and final post-block output. Capture the final declared site only after every residual update and in-place writer contribution.

Attention `o_proj` and MLP `down_proj` can share width but have different input/output semantics. A post-block contrast is not automatically native to either writer. Evaluate candidate writer spaces explicitly rather than assuming one direction is valid everywhere.

## Routed and latent MoE

Logical layers are not equivalent to physical tensor numbers. Sparse experts, shared experts, route masks, gates, and the actual backend loader must agree.

A latent writer may expand `C_latent` to `R`, not residual width `d_model`. Record expert and route coordinates, active token counts, and whether the residual basis is established. Coverage based only on structural enumeration cannot establish decode coverage.

Measure shared expert consumers separately. A shared expert can affect general computation even if routed experts carry most activation mass.

## Recurrent and hybrid networks

Specify the state component, recurrent step, input/output projections, update equations, and nonlinearity placement. Do not treat a complete state vector as a single residual direction or confuse a recurrent update matrix with a matrix that directly writes to that state.

Full-direction modifications can disrupt state transition structure. Candidate low-rank or state-specific operations require new algebraic and behavioral validation.

## Looped and shared-weight networks

Record logical layer, physical module, and loop iteration. A shared tensor can affect multiple sites and iterations. Fused kernels and in-place writes can invalidate ordinary module-hook capture.

v0 rejects shared writers. An adapter supporting them must model all affected consumers and validate the combined update, not edit one name in isolation.

## Multimodal models

Separate language, vision, audio, and projection spaces. A language-side direction cannot be copied into a connector because their dimensions agree. Freeze the intended modality and output contract during comparisons; test cross-modal regression separately.

## Quantized and fused formats

Do not infer codec capabilities from a file extension. Inspect quantization scheme, weight/activation precision, group layout, packing, tensor identity, and backend packing behavior. Qualify decode, applicability, exact supported write operations, readback, serving, and behavior separately.

Safe unquantized `safetensors` is the only v0 reference codec. Unsupported formats fail closed; no compressed-tensor patching is provided.

## Negative results to retain

- A zero/null intervention is a result, not a reason to transfer an arbitrary sibling recipe.
- Magnitude rankings are evidence about one measurement, not proof of causal importance.
- Strong behavior change can be nonspecific disruption; test capability and specificity.
- A format round-trip does not demonstrate that the edited artifact serves or behaves acceptably.
- Cross-model transfer requires fresh discovery and qualification, not only dimension matching.
