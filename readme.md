# PromptEnhance

A CLI tool that preprocesses developer prompts by combining deterministic codebase analysis with LLM-powered enhancement. It gathers relevant project context — file contents, symbol definitions, tool output — and assembles an enriched prompt ready for downstream AI agents.

## Usage

```bash
promptenhance [flags] "your prompt here"
```

### Flags

| Flag | Shorthand | Default | Description |
|---|---|---|---|
| `--model` | - | `qwen2.5-coder:7b` (or `$OLLAMA_MODEL`) | Ollama model name to use for prompt enhancement |
| `--verbose` | `-v` | `false` | Print verbose execution logs of the pipeline |
| `--silent` | `-s` | `false` | Omit any text from the output except the final prompt |
| `--max-passes` | - | `2` | Maximum number of pipeline passes for context discovery |
| `--start` | - | `""` | Start TUI session with the enhanced prompt: `claude`, `opencode`, or `agy` |

### Examples

* Run silent mode to capture only the enhanced prompt text:
  ```bash
  promptenhance --silent "Rename user.Name to user.FullName"
  ```
* Run with verbose logging to inspect what tools the LLM is calling:
  ```bash
  promptenhance -v "Fix the broken tests in internal/pipeline"
  ```

## Pipeline Flow

1. **Discovery Pass**: Uses `agentdb` functions to gather a rough context, especially the `locate_issue_impact_area` functionality.
2. **Context Discovery Loop**: Passes the gathered context to a local LLM model. The LLM reviews the prompt and context, and is allowed to call additional `agentdb` functions to narrow down target files or symbols.
3. **Refinement Pass**: The model refines the prompt for accuracy and appends evaluated context metadata (without raw code blocks) to guide downstream agents.