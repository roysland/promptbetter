package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/roysland/localpromptenhance/internal/agentdb"
	"github.com/roysland/localpromptenhance/internal/ollama"
)

// Pipeline orchestrates the prompt enhancement flow.
type Pipeline struct {
	agentdbClient *agentdb.Client
	ollamaClient  *ollama.Client
	Verbose       bool
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

type LLMResponse struct {
	ToolCalls   []ToolCall `json:"tool_calls"`
	Explanation string     `json:"explanation"`
	FinalPrompt string     `json:"final_prompt"`
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
Your job is to read a user's prompt and any codebase context gathered so far, then decide which additional AgentDB tools to call to gather relevant codebase context to understand the issue/request.

You MUST output your response strictly as a JSON object matching this schema:
{
  "tool_calls": [
    {
      "name": "locate_issue_impact_area" | "get_imports" | "get_callers" | "get_callees" | "semantic_search",
      "arguments": {
         // for locate_issue_impact_area: "issue_text" (string, e.g. the original prompt or a query description)
         // for get_imports: "file_path" (string)
         // for get_callers: "name" (string, symbol name)
         // for get_callees: "qualified_name" (string, e.g. "package.Symbol")
         // for semantic_search: "query" (string)
      }
    }
  ],
  "explanation": "Why you are requesting these tools, or why you have enough context",
  "final_prompt": "Only provide this if tool_calls is empty. Do NOT include raw code blocks in final_prompt. The final prompt should describe the issue precisely and point to the relevant files/symbols."
}

Available AgentDB tools are:
1. locate_issue_impact_area (parameters: issue_text)
2. get_imports (parameters: file_path)
3. get_callers (parameters: name)
4. get_callees (parameters: qualified_name)
5. semantic_search (parameters: query)

If you have enough information to refine the prompt without calling more tools, make "tool_calls" empty and supply the "final_prompt".`

	systemPromptFinal := `You are an expert developer prompt preprocessor.
Read the gathered context results. This is the final pass, so you MUST now output your final prompt. You cannot call any more tools.

Include all relevant details from the tool results (such as specific file paths, symbol names, line numbers, or caller relationships) directly in the "final_prompt". This saves downstream agents from having to look up this information again.

Output strictly as a JSON object matching this schema:
{
  "explanation": "Final explanation describing why the prompt is ready",
  "final_prompt": "Your refined prompt here. It MUST describe the problem, point to the target files/symbols, and list specific file paths and line numbers/ranges from the context, but it MUST NOT contain raw code blocks."
}`

	// Initialize message history
	messages := []ollama.Message{
		{Role: "system", Content: systemPromptDiscovery},
		{Role: "user", Content: fmt.Sprintf("User Prompt: %s", rawPrompt)},
	}

	var accumulatedContext []string

	for pass := 1; pass <= maxPasses; pass++ {
		isFinalPass := (pass == maxPasses)

		// Set the appropriate system instruction for this pass
		if isFinalPass {
			messages[0] = ollama.Message{Role: "system", Content: systemPromptFinal}
		} else {
			messages[0] = ollama.Message{Role: "system", Content: systemPromptDiscovery}
		}

		if p.Verbose {
			fmt.Printf("[DEBUG] --- Sending Pass %d Messages to Ollama ---\n", pass)
			for _, m := range messages {
				fmt.Printf("Role: %s\nContent:\n%s\n\n", m.Role, m.Content)
			}
		}

		fmt.Printf("🤖 Running Pass %d through Ollama (%s)...\n", pass, model)
		respText, err := p.ollamaClient.Chat(ctx, model, messages)
		if err != nil {
			return "", fmt.Errorf("ollama pass %d failed: %w", err, pass)
		}

		if p.Verbose {
			fmt.Printf("[DEBUG] --- Pass %d Raw Response from Ollama ---\n%s\n\n", pass, respText)
		}

		llmResp, err := parseLLMResponse(respText)
		if err != nil {
			if isFinalPass {
				fmt.Printf("⚠️ Warning: Failed parsing LLM response as JSON in final pass (%v). Using raw response as prompt.\n", err)
				return buildContextmaxPrompt(respText, accumulatedContext), nil
			}
			return "", fmt.Errorf("failed parsing LLM JSON response in pass %d: %w. Raw text: %s", pass, err, respText)
		}

		// If it's the final pass, or the LLM decided it has enough info (no tool calls)
		if isFinalPass || len(llmResp.ToolCalls) == 0 {
			if llmResp.FinalPrompt != "" {
				return buildContextmaxPrompt(llmResp.FinalPrompt, accumulatedContext), nil
			}
			if isFinalPass {
				fmt.Printf("⚠️ Warning: LLM did not output a final prompt in the final pass. Falling back gracefully.\n")
				if llmResp.Explanation != "" {
					fallback := fmt.Sprintf("%s\n\n(Note: LLM context discovery suggested: %s)", rawPrompt, llmResp.Explanation)
					return buildContextmaxPrompt(fallback, accumulatedContext), nil
				}
				return buildContextmaxPrompt(rawPrompt, accumulatedContext), nil
			}
		}

		// Execute tools for intermediate passes
		var gatheredContext []string
		for _, tc := range llmResp.ToolCalls {
			fmt.Printf("⚙️ Executing AgentDB tool %s with arguments %+v...\n", tc.Name, tc.Arguments)
			res, err := p.agentdbClient.CallTool(ctx, tc.Name, tc.Arguments)
			if err != nil {
				fmt.Printf("⚠️ Warning: tool execution %s failed: %v\n", tc.Name, err)
				gatheredContext = append(gatheredContext, fmt.Sprintf("Tool %s failed: %v", tc.Name, err))
			} else {
				if p.Verbose {
					fmt.Printf("[DEBUG] Tool %s output:\n%s\n\n", tc.Name, res)
				}
				gatheredResult := fmt.Sprintf("Results from tool %s (args: %+v):\n%s", tc.Name, tc.Arguments, res)
				gatheredContext = append(gatheredContext, gatheredResult)

				// Keep a markdown copy for the contextmax prompt suffix
				accumulatedContext = append(accumulatedContext, fmt.Sprintf("### Tool: %s (arguments: %+v)\n```json\n%s\n```", tc.Name, tc.Arguments, res))
			}
		}

		// Append the assistant response and the tool results user message to history
		messages = append(messages, ollama.Message{Role: "assistant", Content: respText})
		messages = append(messages, ollama.Message{
			Role:    "user",
			Content: fmt.Sprintf("Here is the requested additional tool context from Pass %d:\n\n%s", pass, strings.Join(gatheredContext, "\n\n")),
		})
	}

	return "", fmt.Errorf("pipeline ended without producing a prompt")
}

func buildContextmaxPrompt(prompt string, contextList []string) string {
	if len(contextList) == 0 {
		return prompt
	}
	return fmt.Sprintf("%s\n\n---\n⚡ **CONTEXTMAX: Pre-Gathered Codebase Context** ⚡\n\nThis context was pre-gathered via AgentDB tools to assist in solving the issue. Do not query for this metadata again unless needed.\n\n%s", prompt, strings.Join(contextList, "\n\n"))
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
