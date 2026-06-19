# GFAz documentation

This directory holds the project documentation. The top-level
[`README.md`](../README.md) is the entry point; the files here go deeper.

## Build

- [BUILD_GUIDE.md](BUILD_GUIDE.md) — build instructions, CMake options
  (`ENABLE_CUDA`, `BUILD_CLI`, `BUILD_PYTHON_BINDINGS`, `GFAZ_USE_SYSTEM_ZSTD`,
  …), dependencies, and troubleshooting.

## Internals

- [WORKFLOW.md](WORKFLOW.md) — CPU/GPU compression & decompression workflows and
  the shared `.gfaz` serialization contract (magic `GFAZ`, version 5).

## Compute-engine workflows (`reference` specs for shipped subcommands)

- [workflows/GROWTH_WORKFLOW.md](workflows/GROWTH_WORKFLOW.md) — `gfaz growth`,
  the Panacus-equivalent pangenome growth curve.
- [workflows/PAV_WORKFLOW.md](workflows/PAV_WORKFLOW.md) — `gfaz pav`, the
  odgi-equivalent presence/absence ratio computation over BED ranges.
- [workflows/DECONSTRUCT_WORKFLOW.md](workflows/DECONSTRUCT_WORKFLOW.md) —
  `gfaz deconstruct`, the vg-equivalent GFA→VCF workflow.

## Design & roadmap (forward-looking; not user manuals)

- [design/COMPUTE_ENGINE_DIRECTION.md](design/COMPUTE_ENGINE_DIRECTION.md) —
  strategic direction for GFAz as a compressed pangenome compute engine.
- [design/DOWNSTREAM_APPLICATIONS.md](design/DOWNSTREAM_APPLICATIONS.md) —
  taxonomy of downstream analytics and their priority.
- [design/VCF_future_plan.md](design/VCF_future_plan.md) — remaining VCF-related
  roadmap (deconstruct is shipped; genotype/AF export, VCF codec, construct, …).
- [design/gpu_backend_optimization_plan.md](design/gpu_backend_optimization_plan.md)
  — active plan for GPU correctness/perf work (GPU is experimental).

Older, superseded notes live in [`../deprecated/`](../deprecated/).
