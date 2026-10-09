# R-Judge collection settings

Requests were collected through OpenRouter on October 6, 2026, from 21:17 to 21:35 UTC. Seven reasoning variants were collected; the document and frozen figure inputs show the following five. The omitted variants are GPT-5 (low) and GPT-5 Nano (low).

| Figure label | Requested model | Frozen canonical slug | Requested reasoning effort | Pinned provider tag | Requested max_tokens |
|---|---|---|---|---|---:|
| GPT-5 Nano (medium) | openai/gpt-5-nano | openai/gpt-5-nano-2025-08-07 | medium | openai | 7680 |
| GPT-5 (medium) | openai/gpt-5 | openai/gpt-5-2025-08-07 | medium | openai/default | 7680 |
| GPT-6 Astra (high) | openai/gpt-6-astra | openai/gpt-6-astra-20260903 | high | openai | 7680 |
| Gemini 3.8 Flash (high) | google/gemini-3.8-flash | google/gemini-3.8-flash-20260902 | high | google-ai-studio | 15360 |
| Opus 4.8 (high) | anthropic/claude-opus-4.8 | anthropic/claude-4.8-opus-20260528 | high | anthropic | 7680 |

Every request used one user message, `response_format={"type":"json_object"}`, `service_tier="default"`, and `stream=false`. Provider fallback was disabled and supported parameters were required. No temperature, top-p, or API seed was supplied. Reasoning effort describes the requested API setting. Gemini's 65,536-token allowance was a conservative cost reservation, not its requested output cap.

The exact rubric is in [rjudge_prompt.txt](rjudge_prompt.txt). The prompt inserted only the source agent profile and complete transcript; it excluded source labels, risk descriptions, attack types, scenario, goal, and record ID. It asked for a reason and a 0–10 score representing confidence that agent behavior was unsafe, not harm severity. It expressly treated a risky action without confirmation as unsafe even when the user requested that action. One complete source record received one score, with no chunking or aggregation.

The pinned R-Judge revision is `83ce301da3ad50dd8b397e772863f5411c3d3dc2`. All 571 published records were retained, including duplicate inputs: 301 unsafe and 270 safe. Each displayed model has 571 valid scores except Gemini and Opus, which each have 570 (300 unsafe and 270 safe). Missing scores were excluded, never set to zero. The first valid response per record and variant was selected. Operating points interpolate empirical ROC vertices and can require random selection among tied scores; they are descriptive benchmark results, not thresholds validated on a held-out sample.

These settings were checked against the original frozen `modified_data/rjudge_requests.jsonl` and `modified_data/rjudge_requests_manifest.json`, rather than inferred from current API defaults. The original request-manifest SHA-256 is `6afb274fa0cb947c4be096ac1d959f8b84e3b8f49e24f6e29f4466485a2718f3`; the rubric SHA-256 is `c1e6cb1ad116e9a081ba60aed335035175ec22a4baf003ba337f84cd28dfc122`. Collection scripts and instantiated requests are not required to redraw the frozen figures, and are not included here.
