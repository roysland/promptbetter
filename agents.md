The goal is to enhance FIRST PROMPT accuracy and context when starting a new agentic session.
The main philosophy is to enhance the usage of local LLM models, so any third party call is strictly out of scope.
We only handle ollama as endpoint. 

This first prompt enhancer will leverage the agentdb tool. Therefore, agentdb is a hard requirement. It will not run without agentdb in path, or in a directory that is not the root directory of an agentdb registered project.

This is a CLI. 
Usage:
```
promptenhance "Figure out why the <dialog> doesn't open when clicking the button#dialog1"
```

It supports starting a TUI session with the improved prompt, by passing
--start claude 
--start opencode
--start agy

You can pick model directly by passing "--model <modelname>" into the command. 

This project is a scaled down version of the now abandoned prompt-enhance (~/Projects/System/prompt-enhance)
There is a lot of good code and patterns that can be reused. Use agentdb to lookup the codebase.

