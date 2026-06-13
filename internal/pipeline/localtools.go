package pipeline

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// runGitCommand helper runs a git command in the project root.
func runGitCommand(projectRoot string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = projectRoot
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git command failed: %w (output: %q)", err, string(out))
	}
	return string(out), nil
}

// executeGitDiff gets the current git diff of unstaged/staged changes, truncated to a safe size.
func executeGitDiff(projectRoot string) (string, error) {
	// Verify git repo
	if _, err := os.Stat(filepath.Join(projectRoot, ".git")); os.IsNotExist(err) {
		return "Not a git repository (no .git folder found).", nil
	}

	diff, err := runGitCommand(projectRoot, "diff")
	if err != nil {
		return "", err
	}

	if strings.TrimSpace(diff) == "" {
		return "No changes (git diff is empty).", nil
	}

	// Truncate to prevent context window explosion
	const maxDiffLen = 15000 // approx 3-4k tokens
	if len(diff) > maxDiffLen {
		return diff[:maxDiffLen] + "\n\n[... Git diff truncated for size ...]", nil
	}
	return diff, nil
}

// executeGitStatus gets the git status in short form.
func executeGitStatus(projectRoot string) (string, error) {
	if _, err := os.Stat(filepath.Join(projectRoot, ".git")); os.IsNotExist(err) {
		return "Not a git repository.", nil
	}

	status, err := runGitCommand(projectRoot, "status", "--short")
	if err != nil {
		return "", err
	}

	if strings.TrimSpace(status) == "" {
		return "Working directory clean.", nil
	}
	return status, nil
}

// executeReadFile reads a local file relative to the project root, capped at a maximum line/size limit.
func executeReadFile(projectRoot, filePath string) (string, error) {
	// Secure path resolution relative to projectRoot
	targetPath := filepath.Clean(filePath)
	if !filepath.IsAbs(targetPath) {
		targetPath = filepath.Join(projectRoot, targetPath)
	}

	// Ensure targetPath does not escape projectRoot
	rel, err := filepath.Rel(projectRoot, targetPath)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("security error: path %q escapes project root", filePath)
	}

	file, err := os.Open(targetPath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	// Capped read
	const maxBytes = 30000 // approx 7-8k tokens max
	buf := make([]byte, maxBytes)
	n, err := file.Read(buf)
	if err != nil && err != io.EOF {
		return "", err
	}

	content := string(buf[:n])
	
	// Check if file was truncated
	fi, err := file.Stat()
	if err == nil && fi.Size() > int64(maxBytes) {
		content += "\n\n[... File truncated for size limit ...]"
	}

	return content, nil
}

// executeListDirectory lists files and folders inside dirPath.
func executeListDirectory(projectRoot, dirPath string) (string, error) {
	targetPath := filepath.Clean(dirPath)
	if !filepath.IsAbs(targetPath) {
		targetPath = filepath.Join(projectRoot, targetPath)
	}

	rel, err := filepath.Rel(projectRoot, targetPath)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("security error: path %q escapes project root", dirPath)
	}

	entries, err := os.ReadDir(targetPath)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	for _, entry := range entries {
		info, err := entry.Info()
		size := int64(0)
		if err == nil {
			size = info.Size()
		}

		if entry.IsDir() {
			sb.WriteString(fmt.Sprintf("[DIR]  %s/\n", entry.Name()))
		} else {
			sb.WriteString(fmt.Sprintf("[FILE] %s (%d bytes)\n", entry.Name(), size))
		}
	}

	result := sb.String()
	if result == "" {
		return "Directory is empty.", nil
	}
	return result, nil
}
