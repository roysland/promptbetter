package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/roysland/localpromptenhance/internal/agentdb"
	"github.com/roysland/localpromptenhance/internal/launcher"
	"github.com/roysland/localpromptenhance/internal/ollama"
	"github.com/roysland/localpromptenhance/internal/pipeline"
)

func main() {
	// 1. Determine default model
	defaultModel := os.Getenv("OLLAMA_MODEL")
	if defaultModel == "" {
		defaultModel = "qwen2.5-coder:7b"
	}

	// 2. Define flags
	modelFlag := flag.String("model", defaultModel, "Ollama model name to use for prompt enhancement")
	startFlag := flag.String("start", "", "Start TUI session with the enhanced prompt: 'claude', 'opencode', or 'agy'")
	
	var verbose bool
	flag.BoolVar(&verbose, "verbose", false, "Print verbose execution logs of the pipeline")
	flag.BoolVar(&verbose, "v", false, "Print verbose execution logs of the pipeline (shorthand)")
	
	var silent bool
	flag.BoolVar(&silent, "silent", false, "Omit any text from the output except the final prompt")
	flag.BoolVar(&silent, "s", false, "Omit any text from the output except the final prompt (shorthand)")

	maxPassesFlag := flag.Int("max-passes", 2, "Maximum number of pipeline passes for context discovery")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: promptenhance [flags] \"your prompt here\"\n\nFlags:\n")
		flag.PrintDefaults()
	}

	flag.Parse()

	// 3. Extract positional argument (prompt)
	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}
	rawPrompt := strings.Join(args, " ")
	if strings.TrimSpace(rawPrompt) == "" {
		flag.Usage()
		os.Exit(1)
	}

	// 4. Verify agentdb is in PATH
	if _, err := exec.LookPath("agentdb"); err != nil {
		fmt.Fprintln(os.Stderr, "❌ Error: 'agentdb' binary not found in PATH. It is a hard requirement for prompt enhancement.")
		os.Exit(1)
	}

	// 5. Get current working directory (project root)
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Error getting current working directory: %v\n", err)
		os.Exit(1)
	}
	absRoot, err := filepath.Abs(cwd)
	if err != nil {
		absRoot = cwd
	}

	// 6. Initialize AgentDB Client
	dbClient, err := agentdb.NewClient(absRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Error initializing AgentDB client: %v\n", err)
		fmt.Fprintln(os.Stderr, "Please ensure you are running this command from the root of an AgentDB registered project.")
		os.Exit(1)
	}
	defer dbClient.Close()

	// 7. Initialize Ollama Client
	ollClient := ollama.NewClient()

	// 8. Validate Ollama model exists
	ctx := context.Background()
	if err := ollClient.ValidateModel(ctx, *modelFlag); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Error validating Ollama model: %v\n", err)
		os.Exit(1)
	}

	// 9. Initialize Pipeline and run enhancement
	pipe := pipeline.NewPipeline(dbClient, ollClient)
	pipe.Verbose = verbose && !silent
	pipe.Silent = silent

	enhancedPrompt, err := pipe.Enhance(ctx, *modelFlag, rawPrompt, *maxPassesFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Error enhancing prompt: %v\n", err)
		os.Exit(1)
	}

	// 9. Output/Start Session
	if *startFlag != "" {
		sessionType := strings.ToLower(strings.TrimSpace(*startFlag))
		err := launcher.StartSession(sessionType, enhancedPrompt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ Error starting downstream session: %v\n", err)
			os.Exit(1)
		}
	} else {
		if silent {
			fmt.Println(enhancedPrompt)
		} else {
			fmt.Println("\n --- Enhanced Prompt ---")
			fmt.Println(enhancedPrompt)
			fmt.Println("-------------------------")
		}
	}
}
