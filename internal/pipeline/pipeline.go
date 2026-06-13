package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/roysland/localpromptenhance/internal/agentdb"
	"github.com/roysland/localpromptenhance/internal/ollama"
)

// Pipeline orchestrates the prompt enhancement flow.
type Pipeline struct {
	agentdbClient *agentdb.Client
	ollamaClient  *ollama.Client
	Verbose       bool
	Silent        bool
}

// NewPipeline creates a new pipeline instance.
func NewPipeline(db *agentdb.Client, oll *ollama.Client) *Pipeline {
	return &Pipeline{
		agentdbClient: db,
		ollamaClient:  oll,
	}
}

type ToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

type RelevantFile struct {
	FilePath    string   `json:"file_path"`
	Symbols     []string `json:"symbols"`
	Description string   `json:"description"`
}

type LLMResponse struct {
	ToolCalls     []ToolCall     `json:"tool_calls"`
	Explanation   string         `json:"explanation"`
	FinalPrompt   string         `json:"final_prompt"`
	RelevantFiles []RelevantFile `json:"relevant_files"`
}

// Enhance runs context gathering loop and prompt enhancement up to maxPasses.
func (p *Pipeline) Enhance(ctx context.Context, model string, rawPrompt string, maxPasses int) (string, error) {
	if maxPasses < 1 {
		maxPasses = 1
	}

	if p.Verbose {
		fmt.Printf("\n[DEBUG] === START PIPELINE (Max Passes: %d) ===\n[DEBUG] Raw Prompt: %s\n\n", maxPasses, rawPrompt)
	}

	// Define system prompts
	systemPromptDiscovery := `You are an expert developer prompt preprocessor.
Your job is to read a user's prompt and any codebase context gathered so far, then decide which tools to call to gather relevant codebase context to understand the issue/request.

You MUST output your response strictly as a JSON object matching this schema:
{
  "tool_calls": [
    {
      "name": "locate_issue_impact_area" | "get_imports" | "get_callers" | "get_callees" | "semantic_search" | "read_file" | "git_diff" | "git_status" | "list_directory" | "search" | "find_symbol",
      "arguments": {
         // for locate_issue_impact_area: "issue_text" (string)
         // for get_imports: "file_path" (string)
         // for get_callers: "name" (string, symbol name)
         // for get_callees: "qualified_name" (string, e.g. "package.Symbol")
         // for semantic_search: "query" (string)
         // for read_file: "file_path" (string)
         // for git_diff: {} (empty)
         // for git_status: {} (empty)
         // for list_directory: "dir_path" (string)
         // for search: "query" (string)
         // for find_symbol: "name" (string)
      }
    }
  ],
  "explanation": "Why you are requesting these tools, or why you have enough context",
  "final_prompt": "Only provide this if tool_calls is empty. Do NOT include raw code blocks in final_prompt. The final prompt should describe the issue precisely and point to the relevant files/symbols."
}

Available tools:
1. locate_issue_impact_area (parameters: issue_text) - Find relevant code files & symbols using AgentDB.
2. get_imports (parameters: file_path) - List imports of a file using AgentDB.
3. get_callers (parameters: name) - Find callers of a symbol name using AgentDB.
4. get_callees (parameters: qualified_name) - Find outbound calls of a symbol using AgentDB.
5. semantic_search (parameters: query) - Conceptually search code chunks using AgentDB.
6. read_file (parameters: file_path) - Read the raw text of a local file. Useful to inspect actual code logic.
7. git_diff (no parameters) - Fetch unstaged/staged git changes in the workspace. Great to capture current active changes.
8. git_status (no parameters) - View short git status to see modified files.
9. list_directory (parameters: dir_path) - List contents of a local directory.
10. search (parameters: query) - Search for occurrences of strings/text/code snippets using AgentDB. Excellent for finding specific flag usages like "--verbose" or names in code.
11. find_symbol (parameters: name) - Locate exact symbol definitions by name (like struct, variable, or func name) using AgentDB.

CRITICAL RULES FOR TOOL CALLS:
- Never guess file paths or symbol names.
- Never output placeholders like "<current_file>", "<current_function>", or similar bracketed variables.
- If you do not know a file path or symbol name, run "locate_issue_impact_area" or "semantic_search" first to discover them.

If you have enough information to refine the prompt without calling more tools, make "tool_calls" empty and supply the "final_prompt".`

	systemPromptFinal := `You are an expert developer prompt preprocessor.
Read the gathered context results. This is the final pass, so you MUST now output your final prompt. Evaluate the context carefully, filtering out any chatty or low-confidence/irrelevant candidates from AgentDB. Only identify files and symbols that are directly relevant to solving the user request.

Output strictly as a JSON object matching this schema:
{
  "explanation": "Final explanation describing why the prompt is ready",
  "final_prompt": "Your refined prompt here. It MUST describe the problem, point to the target files/symbols clearly, but it MUST NOT contain raw code blocks.",
  "relevant_files": [
    {
      "file_path": "path/to/file.go",
      "symbols": ["Symbol1", "Symbol2"],
      "description": "Brief note on why this file/symbol is relevant"
    }
  ]
}

CRITICAL RULES FOR THE FINAL PROMPT:
- You MUST preserve the core intent, original questions, and specific reasoning requests (such as asking for pros/cons, comparisons, or conceptual explanations) from the user's original prompt. Do not reduce deep design or conceptual queries to a simple coding task.
- Ensure that you do not invent symbol names or hallucinate file paths.`

	// Initialize message history
	messages := []ollama.Message{
		{Role: "system", Content: systemPromptDiscovery},
		{Role: "user", Content: fmt.Sprintf("User Prompt: %s", rawPrompt)},
	}

	// 1. Discovery Loop (passes 1 to maxPasses - 1)
	for pass := 1; pass < maxPasses; pass++ {
		messages[0] = ollama.Message{Role: "system", Content: systemPromptDiscovery}

		if p.Verbose {
			fmt.Printf("[DEBUG] --- Sending Discovery Pass %d Messages to Ollama ---\n", pass)
			for _, m := range messages {
				fmt.Printf("Role: %s\nContent:\n%s\n\n", m.Role, m.Content)
			}
		}

		if !p.Silent {
			fmt.Printf("Running Discovery Pass %d through Ollama (%s)...\n", pass, model)
		}
		respText, err := p.ollamaClient.Chat(ctx, model, messages)
		if err != nil {
			return "", fmt.Errorf("ollama discovery pass %d failed: %w", pass, err)
		}

		if p.Verbose {
			fmt.Printf("[DEBUG] --- Discovery Pass %d Raw Response from Ollama ---\n%s\n\n", pass, respText)
		}

		llmResp, err := parseLLMResponse(respText)
		if err != nil {
			if !p.Silent {
				fmt.Fprintf(os.Stderr, "Warning: Failed parsing LLM response as JSON in discovery pass %d (%v). Using raw response as prompt.\n", pass, err)
			}
			return respText, nil
		}

		// If the LLM decided it has enough info (no tool calls)
		if len(llmResp.ToolCalls) == 0 {
			if llmResp.FinalPrompt != "" {
				return appendCodebaseContext(llmResp.FinalPrompt, llmResp.RelevantFiles), nil
			}
			// Skip remaining discovery passes to run final pass
			break
		}

		// Execute tools for discovery passes
		var gatheredContext []string
		for _, tc := range llmResp.ToolCalls {
			var res string
			var err error

			switch tc.Name {
			case "read_file":
				filePath, _ := tc.Arguments["file_path"].(string)
				if !p.Silent {
					fmt.Printf("Executing internal tool read_file for %q...\n", filePath)
				}
				res, err = executeReadFile(p.agentdbClient.ProjectRoot, filePath)
			case "git_diff":
				if !p.Silent {
					fmt.Printf("Executing internal tool git_diff...\n")
				}
				res, err = executeGitDiff(p.agentdbClient.ProjectRoot)
			case "git_status":
				if !p.Silent {
					fmt.Printf("Executing internal tool git_status...\n")
				}
				res, err = executeGitStatus(p.agentdbClient.ProjectRoot)
			case "list_directory":
				dirPath, _ := tc.Arguments["dir_path"].(string)
				if !p.Silent {
					fmt.Printf("Executing internal tool list_directory for %q...\n", dirPath)
				}
				res, err = executeListDirectory(p.agentdbClient.ProjectRoot, dirPath)
			default:
				if !p.Silent {
					fmt.Printf("Executing AgentDB tool %s with arguments %+v...\n", tc.Name, tc.Arguments)
				}
				res, err = p.agentdbClient.CallTool(ctx, tc.Name, tc.Arguments)
			}

			if err != nil {
				if !p.Silent {
					fmt.Fprintf(os.Stderr, "Warning: tool execution %s failed: %v\n", tc.Name, err)
				}
				gatheredContext = append(gatheredContext, fmt.Sprintf("Tool %s failed: %v", tc.Name, err))
			} else {
				if p.Verbose {
					fmt.Printf("[DEBUG] Tool %s output:\n%s\n\n", tc.Name, res)
				}
				formattedRes := formatToolOutput(tc.Name, res)
				gatheredResult := fmt.Sprintf("Results from tool %s (args: %+v):\n%s", tc.Name, tc.Arguments, formattedRes)
				gatheredContext = append(gatheredContext, gatheredResult)
			}
		}

		// Append assistant choice and tool results to history for next pass
		messages = append(messages, ollama.Message{Role: "assistant", Content: respText})
		messages = append(messages, ollama.Message{
			Role:    "user",
			Content: fmt.Sprintf("Here is the requested additional context from discovery pass %d:\n\n%s", pass, strings.Join(gatheredContext, "\n\n")),
		})
	}

	// 2. Final Pass (Always executes exactly once at the end)
	messages[0] = ollama.Message{Role: "system", Content: systemPromptFinal}

	if p.Verbose {
		fmt.Printf("[DEBUG] --- Sending Final Pass Messages to Ollama ---\n")
		for _, m := range messages {
			fmt.Printf("Role: %s\nContent:\n%s\n\n", m.Role, m.Content)
		}
	}

	if !p.Silent {
		fmt.Printf("Running Final Pass through Ollama (%s)...\n", model)
	}
	finalRespText, err := p.ollamaClient.Chat(ctx, model, messages)
	if err != nil {
		return "", fmt.Errorf("ollama final pass failed: %w", err)
	}

	if p.Verbose {
		fmt.Printf("[DEBUG] --- Final Pass Raw Response from Ollama ---\n%s\n\n", finalRespText)
	}

	llmResp, err := parseLLMResponse(finalRespText)
	if err != nil {
		if !p.Silent {
			fmt.Fprintf(os.Stderr, "Warning: Failed parsing LLM response as JSON in final pass (%v). Using raw response as prompt.\n", err)
		}
		return finalRespText, nil
	}

	if llmResp.FinalPrompt != "" {
		return appendCodebaseContext(llmResp.FinalPrompt, llmResp.RelevantFiles), nil
	}

	if !p.Silent {
		fmt.Fprintf(os.Stderr, "Warning: LLM did not output a final prompt in the final pass. Falling back gracefully.\n")
	}
	if llmResp.Explanation != "" {
		fallback := fmt.Sprintf("%s\n\n(Note: LLM context discovery suggested: %s)", rawPrompt, llmResp.Explanation)
		return appendCodebaseContext(fallback, llmResp.RelevantFiles), nil
	}
	return appendCodebaseContext(rawPrompt, llmResp.RelevantFiles), nil
}

func appendCodebaseContext(prompt string, relevantFiles []RelevantFile) string {
	if len(relevantFiles) == 0 {
		return prompt
	}
	var sb strings.Builder
	sb.WriteString(prompt)
	sb.WriteString("\n\n---\nCONTEXTMAX: Evaluated Codebase Context\n\nThe following files and symbols were evaluated by the preprocessing LLM and determined to be relevant to the issue:\n\n")
	for _, rf := range relevantFiles {
		sb.WriteString(fmt.Sprintf("- **%s**", rf.FilePath))
		if len(rf.Symbols) > 0 {
			sb.WriteString(fmt.Sprintf(" (Symbols: `%s`)", strings.Join(rf.Symbols, "`, `")))
		}
		sb.WriteString("\n")
		if rf.Description != "" {
			sb.WriteString(fmt.Sprintf("  * Note: %s\n", rf.Description))
		}
	}
	return sb.String()
}

func parseLLMResponse(raw string) (LLMResponse, error) {
	// Strip markdown blocks if model wraps json in ```json ... ```
	clean := raw
	if idx := strings.Index(clean, "```json"); idx != -1 {
		clean = clean[idx+7:]
		if endIdx := strings.LastIndex(clean, "```"); endIdx != -1 {
			clean = clean[:endIdx]
		}
	} else if idx := strings.Index(clean, "```"); idx != -1 {
		clean = clean[idx+3:]
		if endIdx := strings.LastIndex(clean, "```"); endIdx != -1 {
			clean = clean[:endIdx]
		}
	}

	clean = strings.TrimSpace(clean)

	var resp LLMResponse
	if err := json.Unmarshal([]byte(clean), &resp); err != nil {
		return resp, err
	}
	return resp, nil
}

func formatToolOutput(toolName string, rawOutput string) string {
	rawOutput = strings.TrimSpace(rawOutput)
	if rawOutput == "" {
		return "No output."
	}

	switch toolName {
	case "locate_issue_impact_area":
		var data struct {
			Candidates []struct {
				Symbol struct {
					FilePath  string `json:"file_path"`
					Name      string `json:"name"`
					StartLine int    `json:"start_line"`
					EndLine   int    `json:"end_line"`
				} `json:"symbol"`
				ConfidenceScore float64 `json:"confidence_score"`
				Chunks          []struct {
					Snippet string `json:"snippet"`
				} `json:"chunks"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal([]byte(rawOutput), &data); err == nil && len(data.Candidates) > 0 {
			var sb strings.Builder
			for i, c := range data.Candidates {
				sb.WriteString(fmt.Sprintf("[%d] Candidate File: %s | Symbol: %s (Lines: %d-%d) | Confidence: %.0f%%\n",
					i+1, c.Symbol.FilePath, c.Symbol.Name, c.Symbol.StartLine, c.Symbol.EndLine, c.ConfidenceScore*100))
				if len(c.Chunks) > 0 {
					sb.WriteString("Snippet:\n```\n")
					sb.WriteString(strings.TrimSpace(c.Chunks[0].Snippet))
					sb.WriteString("\n```\n")
				}
				sb.WriteString("\n")
			}
			return strings.TrimSpace(sb.String())
		}
	case "search":
		var data struct {
			Results []struct {
				FilePath  string  `json:"file_path"`
				StartLine int     `json:"start_line"`
				EndLine   int     `json:"end_line"`
				Snippet   string  `json:"snippet"`
				Score     float64 `json:"score"`
			} `json:"results"`
		}
		if err := json.Unmarshal([]byte(rawOutput), &data); err == nil && len(data.Results) > 0 {
			var sb strings.Builder
			for i, r := range data.Results {
				sb.WriteString(fmt.Sprintf("[%d] File: %s (Lines: %d-%d) | Score: %.2f\n",
					i+1, r.FilePath, r.StartLine, r.EndLine, r.Score))
				sb.WriteString("Snippet:\n```\n")
				sb.WriteString(strings.TrimSpace(r.Snippet))
				sb.WriteString("\n```\n\n")
			}
			return strings.TrimSpace(sb.String())
		}
	case "semantic_search":
		var data struct {
			Results []struct {
				FilePath   string  `json:"file_path"`
				StartLine  int     `json:"start_line"`
				EndLine    int     `json:"end_line"`
				Name       string  `json:"name"`
				Kind       string  `json:"kind"`
				Signature  string  `json:"signature"`
				DocComment string  `json:"doc_comment"`
				Score      float64 `json:"score"`
			} `json:"results"`
		}
		if err := json.Unmarshal([]byte(rawOutput), &data); err == nil && len(data.Results) > 0 {
			var sb strings.Builder
			for i, r := range data.Results {
				sb.WriteString(fmt.Sprintf("[%d] Symbol: %s (%s) in %s (Lines: %d-%d) | Score: %.2f\n",
					i+1, r.Name, r.Kind, r.FilePath, r.StartLine, r.EndLine, r.Score))
				if r.Signature != "" {
					sb.WriteString(fmt.Sprintf("  Signature: `%s`\n", strings.TrimSpace(r.Signature)))
				}
				if r.DocComment != "" {
					sb.WriteString(fmt.Sprintf("  Doc: %s\n", strings.TrimSpace(r.DocComment)))
				}
				sb.WriteString("\n")
			}
			return strings.TrimSpace(sb.String())
		}
	case "find_symbol":
		var data struct {
			Symbols []struct {
				FilePath   string `json:"file_path"`
				Name       string `json:"name"`
				StartLine  int    `json:"start_line"`
				EndLine    int    `json:"end_line"`
				Signature  string `json:"signature"`
				DocComment string `json:"doc_comment"`
			} `json:"symbols"`
		}
		if err := json.Unmarshal([]byte(rawOutput), &data); err == nil && len(data.Symbols) > 0 {
			var sb strings.Builder
			for i, s := range data.Symbols {
				sb.WriteString(fmt.Sprintf("[%d] Symbol: %s in %s (Lines: %d-%d)\n",
					i+1, s.Name, s.FilePath, s.StartLine, s.EndLine))
				if s.Signature != "" {
					sb.WriteString(fmt.Sprintf("  Signature: `%s`\n", strings.TrimSpace(s.Signature)))
				}
				if s.DocComment != "" {
					sb.WriteString(fmt.Sprintf("  Doc: %s\n", strings.TrimSpace(s.DocComment)))
				}
				sb.WriteString("\n")
			}
			return strings.TrimSpace(sb.String())
		}
	case "get_imports":
		var data struct {
			Imports []string `json:"imports"`
		}
		if err := json.Unmarshal([]byte(rawOutput), &data); err == nil {
			return fmt.Sprintf("Imports:\n- %s", strings.Join(data.Imports, "\n- "))
		}
	case "get_callers", "get_callees":
		var data struct {
			Symbols []string `json:"symbols"`
		}
		if err := json.Unmarshal([]byte(rawOutput), &data); err == nil {
			return fmt.Sprintf("Related Symbols:\n- %s", strings.Join(data.Symbols, "\n- "))
		}
	}

	return rawOutput
}
