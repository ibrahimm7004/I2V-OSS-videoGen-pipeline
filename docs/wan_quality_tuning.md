# WAN Quality Tuning

## Baseline Defaults (Recommended)

Use these defaults for full-quality WAN idea runs:

- `video.width: 1280`
- `video.height: 704` (WAN native 720p path)
- `generation_defaults.chain_last_frame: true`
- `generation_defaults.steps: 60`
- `generation_defaults.guidance_scale: 5.0`
- `generation_defaults.motion_strength: 0.70`
- `generation_defaults.export_quality: 9`
- `generation_defaults.wan_profile: quality`
- `generation_defaults.cinematic_constraints: true`

For smoke runs:

- `generation_defaults.wan_profile: smoke`
- `generation_defaults.cinematic_constraints: false` (default behavior for smoke if unset)
- `generation_defaults.motion_block: false` (default behavior for smoke if unset)

## Troubleshooting

| Symptom | Likely Cause | Recommended Fix |
|---|---|---|
| Output looks soft/blurred | Too few steps, weak detail constraints, low encode quality | Increase `steps` to `60-72`, keep `export_quality=9`, keep cinematic constraints enabled |
| Face/identity drift between clips | Chaining disabled or weak continuity prompting | Set `chain_last_frame=true`, keep continuity rules explicit, keep cinematic constraints enabled |
| Output too static | Motion too low or over-conditioning | Increase `motion_strength` by `+0.1` (up to `0.85`), lower guidance by `-0.3` (floor `4.5`) |
| Output too chaotic | Motion too high or guidance too low | Lower `motion_strength` toward `0.60-0.70`, raise guidance toward `5.0-5.8` |
| Aggressive push-in/zoom | Prompt lacks camera constraints | Keep cinematic constraints enabled; explicitly request stable camera in shot prompt |
| End-frame degradation | Long sequence instability, insufficient steps | Use `steps>=60`, ensure chaining enabled, consider slight guidance increase if details collapse |
| Compression artifacts/blockiness | Encode quality/bitrate too low | Keep `export_quality=9`; optionally enable postprocess FFmpeg re-encode with lower CRF |

## Cinematic Constraints Block

The runner appends this block to WAN clip prompts when enabled:

```text
[CINEMATIC_CONSTRAINTS]
- slow subtle dolly-in only
- no rapid zoom
- no sudden framing changes
- subject centered
- face identity consistent
- crisp focus on subject
- fine fabric detail
- no smear
- 35mm lens
- filmic contrast
```

Disable globally:

```yaml
generation_defaults:
  cinematic_constraints: false
```

Disable per shot:

```yaml
shots:
  - index: 2
    prompt: "..."
    cinematic_constraints: false
```

## Motion blocks

Motion blocks are per-shot motion directives injected into prompts to avoid still-looking clips with only camera drift.

Defaults:

- WAN `quality`: `generation_defaults.motion_block: true`
- WAN `smoke`: `generation_defaults.motion_block: false`

Per-shot fields (job pack `shots[]`):

- `motion_subject`
- `motion_environment`
- `motion_camera`
- `motion_notes` (optional)
- `motion_block` (per-shot on/off override)

When enabled, the prompt includes:

```text
MOTION:
- Subject: <motion_subject>
- Environment: <motion_environment>
- Camera: <motion_camera>
- Notes: <motion_notes>   # only if non-empty
```

Disable globally:

```yaml
generation_defaults:
  motion_block: false
```

Disable per shot:

```yaml
shots:
  - index: 3
    prompt: "..."
    motion_block: false
```

## Adaptive Retry (WAN-only)

When a clip fails validation with one of these signals, the pipeline retries once:

- static output failure
- encoded frame mismatch
- MP4 too small

Retry adjustments:

- `motion_strength += 0.10` (cap `0.85`)
- `guidance_scale -= 0.3` (floor `4.5`)
- `steps += 6` (cap `72`)

Retry metadata is recorded per clip in `logs/log_XXX.json` under `adapter_metadata`.

## Postprocess (Optional, Off by Default)

Postprocess is optional and defaults to disabled:

```yaml
generation_defaults:
  postprocess:
    enable: false
```

Available knobs (if configured):

- `postprocess.ffmpeg_crf`
- `postprocess.preset`
- `postprocess.tune`
- `postprocess.bitrate`
- `postprocess.sharpen`
- optional upscaler binary/settings

If postprocess dependencies are missing, the pipeline logs a warning and continues.
