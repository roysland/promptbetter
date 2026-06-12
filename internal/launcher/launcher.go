package launcher

import (
	"fmt"
	"os"
	"os/exec"
)

// StartSession launches the downstream agent CLI/TUI session.
func StartSession(sessionType string, prompt string) error {
	var cmdName string
	switch sessionType {
	case "claude":
		cmdName = "claude"
	case "opencode":
		cmdName = "opencode"
	case "agy":
		cmdName = "agy"
	default:
		return fmt.Errorf("unsupported downstream agent session type: %q", sessionType)
	}

	// Look for the binary in PATH
	binaryPath, err := exec.LookPath(cmdName)
	if err != nil {
		return fmt.Errorf("could not find binary for downstream session type %q in PATH: %w", cmdName, err)
	}

	fmt.Printf("🚀 Starting downstream TUI session: %s ...\n", cmdName)
	cmd := exec.Command(binaryPath, prompt)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("session failed/exited with error: %w", err)
	}

	return nil
}
