package agentdb

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
)

// Client handles stdio-based MCP communication with agentdb.
type Client struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex
	nextID atomic.Int64

	CodebaseID  int
	ProjectRoot string
	initialized bool
}

type jsonRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int64       `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
}

type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type toolCallParams struct {
	Name      string      `json:"name"`
	Arguments interface{} `json:"arguments"`
}

type mcpToolResult struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content,omitempty"`
}

// NewClient spawns `agentdb mcp`, initializes it, and resolves the codebase ID.
func NewClient(projectRoot string) (*Client, error) {
	cmd := exec.Command("agentdb", "mcp")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	if os.Getenv("PROMPT_ENHANCE_DEBUG") != "" {
		cmd.Stderr = os.Stderr
	} else {
		cmd.Stderr = nil
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start agentdb mcp: %w", err)
	}

	c := &Client{
		cmd:         cmd,
		stdin:       stdin,
		stdout:      bufio.NewReader(stdout),
		ProjectRoot: projectRoot,
	}

	// Handshake
	initParams := map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]interface{}{},
		"clientInfo": map[string]string{
			"name":    "localpromptenhance",
			"version": "0.1.0",
		},
	}
	if _, err := c.call("initialize", initParams); err != nil {
		c.Close()
		return nil, fmt.Errorf("initialize handshake failed: %w", err)
	}

	// Resolve Codebase ID
	if err := c.resolveCodebaseID(); err != nil {
		c.Close()
		return nil, err
	}

	c.initialized = true
	return c, nil
}

// Close terminates the agentdb mcp subprocess.
func (c *Client) Close() error {
	if c.stdin != nil {
		c.stdin.Close()
	}
	if c.cmd != nil && c.cmd.Process != nil {
		return c.cmd.Process.Kill()
	}
	return nil
}

// CallTool calls an MCP tool by name with arguments.
func (c *Client) CallTool(ctx context.Context, toolName string, arguments map[string]interface{}) (string, error) {
	if !c.initialized {
		return "", fmt.Errorf("client not initialized")
	}
	// Inject codebase_id automatically if present in arguments or needed
	if _, ok := arguments["codebase_id"]; !ok && c.CodebaseID != 0 {
		arguments["codebase_id"] = c.CodebaseID
	}

	params := toolCallParams{
		Name:      toolName,
		Arguments: arguments,
	}
	resp, err := c.call("tools/call", params)
	if err != nil {
		return "", err
	}

	var result mcpToolResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return "", fmt.Errorf("unmarshal tool result: %w", err)
	}

	for _, content := range result.Content {
		if content.Type == "text" {
			return content.Text, nil
		}
	}
	return "", fmt.Errorf("no text content in response")
}

func (c *Client) resolveCodebaseID() error {
	params := toolCallParams{
		Name:      "list_codebases",
		Arguments: map[string]interface{}{},
	}
	resp, err := c.call("tools/call", params)
	if err != nil {
		return fmt.Errorf("list_codebases: %w", err)
	}

	var result mcpToolResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return fmt.Errorf("parse list_codebases: %w", err)
	}

	var text string
	for _, content := range result.Content {
		if content.Type == "text" {
			text = content.Text
			break
		}
	}

	var structured struct {
		Items []struct {
			ID       int    `json:"id"`
			RootPath string `json:"root_path"`
		} `json:"items"`
	}
	if err := json.Unmarshal([]byte(text), &structured); err != nil {
		return fmt.Errorf("unmarshal codebase list: %w", err)
	}

	for _, item := range structured.Items {
		if item.RootPath == c.ProjectRoot {
			c.CodebaseID = item.ID
			return nil
		}
	}

	return fmt.Errorf("codebase at path %s is not registered in agentdb", c.ProjectRoot)
}

func (c *Client) call(method string, params interface{}) (*jsonRPCResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	id := c.nextID.Add(1)
	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	data = append(data, '\n')

	if _, err := c.stdin.Write(data); err != nil {
		return nil, fmt.Errorf("write: %w", err)
	}

	line, err := c.stdout.ReadBytes('\n')
	if err != nil {
		if err == io.EOF && len(line) > 0 {
			// use incomplete line
		} else {
			return nil, fmt.Errorf("read line: %w", err)
		}
	}

	var resp jsonRPCResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal: %w (raw: %q)", err, string(line))
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("jsonrpc error (%d): %s", resp.Error.Code, resp.Error.Message)
	}

	return &resp, nil
}
