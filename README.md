# Code Mode for Open WebUI

A single Open WebUI **filter function** that changes how a model uses its tools.
Instead of exposing each enabled toolkit function to the model as a separate
callable, this filter exposes **one** tool, `run_python(code)`, and documents all
the enabled functions as a typed-Python API in the system prompt. The model
writes a short Python program that calls those functions (loops, conditionals,
local filtering, composition), and the filter executes that program in an
in-process sandbox, dispatching each call back to the real Open WebUI tool.

This is the "code mode" pattern, applied inside Open WebUI without touching core
and without adding any container or microVM infrastructure.

## Why an admin might want this

The motivating argument is laid out well in Cloudflare's
[**Code Mode: the better way to use MCP**](https://blog.cloudflare.com/code-mode/).
The short version:

- **LLMs are better at writing code than at emitting tool calls.** Models have
  seen enormous amounts of real-world code in training, but only contrived
  synthetic examples of tool-call formats. Presenting tools as a normal
  programming API plays to the model's strengths, so it can handle more tools,
  and more complex ones, more reliably.
- **Chaining tools stops wasting tokens.** In classic tool calling, the output
  of every call is fed back through the model just to be copied into the next
  call's inputs. When the model writes code instead, it can fan out across many
  calls and only read back the final result it actually needs. Anthropic
  reported large token reductions on tool-heavy flows
  ([Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)).
- **Aggregation happens before context.** The program can filter and summarize
  tool results inside the sandbox, so only the distilled answer reaches the
  model's context window.

The same ideas appear in
[Pydantic's code-mode harness](https://ai.pydantic.dev/code-mode/) and underlie
the interpreter this filter uses, [pydantic-monty](https://github.com/pydantic/monty).

### How this differs from Cloudflare's version

Cloudflare runs each snippet in a fresh, disposable V8 isolate (no state between
runs). This filter runs an in-process Python interpreter and, optionally,
**persists the interpreter state per chat**. That lets the model define a helper
function in one turn and reuse it in a later turn within the same conversation,
effectively letting it author small reusable tools in context. Persistence is a
configurable choice (see Configuration); turn it off for stateless behavior
closer to the Cloudflare model.

## Two ways to deploy it

The function is a per-chat **toggle filter** (`self.toggle = True`), so it
appears as an on/off switch in the chat integrations menu. There are two
deployment postures; pick whichever fits your users.

### Mode 1: Global, opt-in per chat

Install the function and mark it **global**. The toggle then appears for every
chat, and any user can flip it on for a specific conversation when they want
code mode. Nothing changes for chats where the toggle is off.

- Admin → Functions → install `code_mode.py`, make sure it is **active**.
- Toggle the function **global** (in the function's menu, or
  `POST /api/v1/functions/id/<id>/toggle/global`).
- Users see a "Code Mode" toggle in the message-input integrations menu and opt
  in per chat.

This is the lightest-touch option: code mode is available everywhere but never
forced on.

### Mode 2: Per-model, default-on for the models that need it

Leave the function **not global**. Then enable it only on the specific models
that should use code mode, and optionally set it as a default so it is on
without the user doing anything.

- Admin → Functions → install and **activate** the function (not global).
- For each target model: Workspace → Models → the model → enable the Code Mode
  filter, and (optionally) set it as a **default filter** so it is on by default
  for that model.

This keeps the normal admin mental model intact: you still enable toolkits per
model in the usual way, and code mode is just one extra switch on the models
where you want it.

> The two modes are not mutually exclusive in principle, but global + per-model
> defaults can be confusing to reason about. Pick one posture per deployment.

## Prerequisites for a model using code mode

Whichever mode you choose, a model only benefits when:

1. **It has at least one toolkit enabled.** With no tools, the filter passes the
   request through untouched and does nothing.
2. **Function Calling is set to Native.** In default (prompt-based) function
   calling, `body['tools']` is ignored, and the `run_python` tool will not be
   offered correctly. Set the model's function calling mode to **Native**.
3. **Built-in tools are disabled** (`Capabilities → Built-in tools` off, i.e.
   `meta.capabilities.builtin_tools = false`). On native function calling, Open
   WebUI otherwise injects built-in tools that overwrite the single tool this
   filter seeds, and the model's `run_python` call fails with
   `Tool "run_python" not found`.

If any of these are not met, the worst case is that the model behaves as if the
filter were not installed (prerequisite 1) or that `run_python` does not resolve
(prerequisites 2 and 3).

## Configuration

The filter exposes a small set of admin valves:

| Valve | Default | Meaning |
|---|---|---|
| `priority` | `0` | Filter execution order (lower runs first). |
| `persist` | `true` | Keep the Python REPL state per chat, so variables and helper functions the model defines survive across `run_python` calls within a conversation. Turn off for fully stateless execution. |
| `cache_size` | `128` | Maximum number of conversations to retain REPL state for, with LRU eviction. Must be at least 1. Only relevant when `persist` is on. |

State, when persisted, lives in the Open WebUI process memory and is lost on
restart. It is never written to disk and never leaves the host.

## How it works (and why it is unusual)

This filter is intentionally more invasive than a typical filter, so it is worth
understanding before you install it. It performs a few uncommon runtime
substitutions, but nothing that grants the model any capability it did not
already have.

1. **In `inlet`, it reads which toolkits the model enabled** (`tool_ids`) and
   resolves the real Open WebUI tool callables by calling the internal
   `get_tools(...)` helper. This is the one Open WebUI internal that a filter is
   not normally handed. It is a plain importable async function; the filter does
   not monkeypatch anything.
2. **It hides the real tools from the model.** It empties `tool_ids` so Open
   WebUI's own tool resolution produces nothing, then seeds its own single
   `run_python` entry into the execution registry (`metadata['tools']`) and sets
   the model-facing tool list (`body['tools']`) to just `run_python`. The model
   never sees the individual tools as callable functions; it sees them only as
   documentation in the system prompt.
3. **It dispatches sandboxed calls back to the real tools.** When the model
   calls `run_python`, the program runs in
   [pydantic-monty](https://github.com/pydantic/monty), an in-process Python
   interpreter. Each call to a wrapped function pauses the interpreter; the
   filter awaits the real Open WebUI callable and resumes with the result. Tools
   that emit status events or prompt the user interactively work, because Open
   WebUI's own dunder-binding (`__event_emitter__`, `__user__`, etc.) is applied
   to those callables before they are dispatched. The filter forwards the
   standard set of request context and lets Open WebUI bind only what each tool
   declares.

### Security posture

- **No new attack surface from the sandbox.** pydantic-monty is a from-scratch
  interpreter for a subset of Python with no filesystem, network, or environment
  access by construction. The only things the model's program can reach are the
  exact tool functions the toolkit already exposed to it.
- **No new privilege for the model.** The set of callable operations is
  identical to what the enabled toolkits already granted. Code mode changes the
  *shape* of the interface (one code tool over an API), not the *scope* of what
  the model can do.
- **No data spill into logs.** This filter deliberately does **not** log message
  content, submitted code, tool arguments, or tool return values. It emits only
  a few coarse status/warning lines under the `=CODEMODE=` marker. Enabling it
  does not put user data into your container logs that was not already flowing
  there.

### Constraints to be aware of

- **pydantic-monty is alpha.** It implements a subset of Python (`asyncio`,
  `re`, `json`, `datetime`, `os`, `sys`, `typing`; no classes, no third-party
  imports). Programs the model writes must stay in that subset. The version is
  pinned by the function's dependency; treat upgrades as worth re-testing.
- **It uses Open WebUI internals.** The filter imports `get_tools` and writes
  into the native tool-execution registry. This is validated against the
  required Open WebUI version below, but a future core refactor of tool
  resolution could require an update here.

## Installation

1. In Open WebUI: Admin Panel → Functions → **+** → paste the contents of
   [`code_mode.py`](./code_mode.py) (or import the file).
2. **Activate** the function.
3. Choose a deployment mode (global, or per-model default) as described above.
4. Confirm each target model meets the three prerequisites (a toolkit enabled,
   native function calling, built-in tools off).

`pydantic-monty` must be importable in the Open WebUI Python environment. Open
WebUI installs a function's declared requirements automatically; if you run a
locked-down image, ensure `pydantic-monty` is available.

**Required Open WebUI version:** 0.9.6 or compatible (the filter relies on the
native tool-execution loop and the `get_tools` signature from that release).

## License

MIT. See [LICENSE](./LICENSE).
