# Migration: Image Config Option Rename (February 2026)

## What changed

The runtime now uses only canonical image config names:

- `--image-analysis-provider`
- `--image-analysis-model`
- `--image-generation-provider`
- `--image-generation-model`

Runtime env names are also canonical:

- `IMAGE_ANALYSIS_PROVIDER`
- `IMAGE_ANALYSIS_MODEL`
- `IMAGE_GENERATION_PROVIDER`
- `IMAGE_GENERATION_MODEL`

## Removed compatibility names

The following legacy names are no longer accepted:

- CLI flags:
  - `--vision-provider`
  - `--vision-model`
  - `--image-gen-provider`
  - `--image-gen-model`
- Environment variables:
  - `VISION_PROVIDER`
  - `VISION_MODEL`
  - `IMAGE_GEN_PROVIDER`
  - `IMAGE_GEN_MODEL`
  - `VISION_BASE_URL`
  - `VISION_API_KEY`
  - `VISION_TIMEOUT_SECONDS`
  - `IMAGE_GEN_BASE_URL`
  - `IMAGE_GEN_API_KEY`

## Action required

1. Update startup commands to canonical flags.
2. Update shell profiles and deployment secrets to canonical env names.
3. Update any automation/scripts that still use removed names.
