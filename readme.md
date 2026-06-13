# PromptEnhance
zA CLI tool that preprocesses developer prompts by combining deterministic codebase analysis with LLM-powered enhancement. It gathers relevant project context — file contents, symbol definitions, tool output — and assembles an enriched prompt ready for downstream AI agents.

# Pipeline
Use agentdb functions to gather a rough context, especially the locate issue functionality.
It then passes this to a local LLM model. The LLM models reviews the prompt and context given, and is allowed to call additional agentdb functions to narrow down the context.
More context is gathered through agentdb and passed to LLM for a second pass.
LLM agent refines the prompt for accuracy. 

The prompt should not include code itself, only point to the problem. 