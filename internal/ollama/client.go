package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is the request body for Ollama's /api/chat.
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

// ChatResponse is the response body from Ollama's /api/chat.
type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   Message   `json:"message"`
	Done      bool      `json:"done"`
}

// Client is a client for Ollama API.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewClient creates a new Ollama client.
func NewClient() *Client {
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		host := os.Getenv("OLLAMA_HOST")
		if host != "" {
			if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
				baseURL = "http://" + host
			} else {
				baseURL = host
			}
		} else {
			baseURL = "http://localhost:11434"
		}
	}
	baseURL = strings.TrimSuffix(baseURL, "/")

	return &Client{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 90 * time.Second,
		},
	}
}

// Chat calls the /api/chat endpoint.
func (c *Client) Chat(ctx context.Context, model string, messages []Message) (string, error) {
	url := fmt.Sprintf("%s/api/chat", c.BaseURL)

	reqBody := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
		Options: map[string]interface{}{
			"temperature": 0.2, // low temperature for structured tasks
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return "", fmt.Errorf("create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	return chatResp.Message.Content, nil
}

// ValidateModel checks if the given model name is installed in the local Ollama instance.
func (c *Client) ValidateModel(ctx context.Context, modelName string) error {
	url := fmt.Sprintf("%s/api/tags", c.BaseURL)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create check request: %w", err)
	}

	// Short timeout for validation
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("Ollama connection failed: %w (is Ollama running?)", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code from Ollama api/tags: %d", resp.StatusCode)
	}

	var tagsResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		return fmt.Errorf("failed to decode tag list: %w", err)
	}

	var installed []string
	found := false
	for _, m := range tagsResp.Models {
		installed = append(installed, m.Name)
		// Check for exact match or name-prefix match (e.g., "foo" matching "foo:latest")
		if strings.EqualFold(m.Name, modelName) {
			found = true
			break
		}
		if strings.EqualFold(m.Name, modelName+":latest") {
			found = true
			break
		}
		// Also support checking the base name (e.g. user specifies "foo:latest" but tag is "foo")
		if strings.Contains(m.Name, ":") && strings.EqualFold(strings.Split(m.Name, ":")[0], modelName) {
			found = true
			break
		}
	}

	if !found {
		return fmt.Errorf("requested model %q is not installed.\nInstalled models are:\n - %s\nRun 'ollama pull %s' to download it.", modelName, strings.Join(installed, "\n - "), modelName)
	}

	return nil
}
