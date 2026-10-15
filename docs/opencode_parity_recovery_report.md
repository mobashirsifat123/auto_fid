# AutoResearch Recovery Report

## Goal Achieved
The AutoResearch client has been successfully recovered from a divergent custom runtime branch back to a pure, native extension of the upstream OpenCode baseline (`origin/dev`). 

## Key Improvements
1. **Removed Custom Runtime**: The client no longer tries to run a parallel conversational agent loop. It uses the native OpenCode TUI, LLM provider handling, and session state.
2. **Restored File & Terminal Parity**: Normal development commands (e.g., editing files, running bash commands, analyzing diffs) are completely handled by native, mature OpenCode tools rather than custom Python wrappers.
3. **Isolated Research Domain**: The scientific capabilities (Study Search, novelty pipeline, experiment orchestration) remain on the Python backend.
4. **Native Plugin Architecture**: The connection between the CLI and the research backend is now handled purely via the official `@opencode-ai/plugin` interface.
    * The `/discover` command cleanly intercepts via the `command.execute.before` hook.
    * The actual search executes via the `autoresearch_discover` tool, enabling graceful failure if the backend is down without blocking normal coding tasks.

## Validation Status
* Core OpenCode parity suite passed successfully (`bun test`).
* The custom plugin compiles and registers without conflicts.
* `task.md` tracking completed.

## Next Steps for User
1. Verify the client locally by running: `cd ~/autoresearch-cli && bun run --cwd packages/cli --conditions=browser src/index.ts`
2. Test a normal coding query (e.g., "what's the current time using python?").
3. Test a research query using `/discover adaptive basis selection` (ensure the Python backend is running at `http://127.0.0.1:8000`).
