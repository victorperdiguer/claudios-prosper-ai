---
type: "query"
date: "2026-09-23T17:04:49.752360+00:00"
question: "How does the voice engine handle STT, LLM reasoning, tool calls, and TTS?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["AgentReply", "DeepgramEOTCoordinator", "LLMClient", "CallGraph"]
---

# Q: How does the voice engine handle STT, LLM reasoning, tool calls, and TTS?

## Answer

Expanded from graph vocabulary: agent turn deepgram tts llm reply transport. Source analysis is in docs/VOICE_ENGINE_ANALYSIS.md at commit 33bf034f7cce8824984f9aec27ec341f7812ac85. Live path: Twilio PCM transport, Silero VAD, Deepgram streaming STT, custom EOT with local Smart Turn, user aggregator, AgentReply, custom stage tool loop using non-streaming OpenAI-compatible Chat Completions, Cartesia or Deepgram streaming TTS. Code graph excludes semantic docs/config; graph JSON configurations were inspected directly. 111 targeted tests passed. No live provider calls tested.

## Outcome

- Signal: useful

## Source Nodes

- AgentReply
- DeepgramEOTCoordinator
- LLMClient
- CallGraph