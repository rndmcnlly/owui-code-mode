"""
title: Code Mode
author: Adam Smith
author_url: https://github.com/rndmcnlly
version: 1.1.0
license: MIT
description: Replaces a model's individual tool calls with a single run_python(code) tool. The model discovers its enabled toolkit functions as a typed-Python API in the system prompt, then writes a program that calls them. The program runs in an in-process pydantic_monty interpreter; each wrapped-function call is dispatched to the real Open WebUI callable (event-emitting and interactive tools included). REPL state can persist per chat so globals and helper functions the model defines survive across run_python calls within a conversation.
required_open_webui_version: 0.9.6
"""

import json
from collections import OrderedDict
from typing import Optional

from pydantic import BaseModel, Field

from loguru import logger as log

from open_webui.models.users import Users
from open_webui.utils.tools import get_tools

import pydantic_monty


# A greppable marker for the few log lines this filter emits. Note: this filter
# deliberately NEVER logs message content, submitted code, tool arguments, or
# tool return values. Enabling it does not spill user data into container logs
# that was not already flowing there.
MARK = "=CODEMODE="


# ── Per-chat REPL persistence ─────────────────────────────────────────────
#
# Each conversation's interpreter state is held as SERIALIZED BYTES (not a live
# object) in a process-level LRU cache keyed by chat_id. pydantic_monty's
# MontyRepl is a no-replay incremental interpreter: repl.dump() -> bytes and
# MontyRepl.load(bytes) -> repl round-trip its heap and namespace. Each
# run_python call rehydrates from the stored blob, feeds the snippet, then dumps
# the new state back, so a global or function bound in one call is visible in
# the next call OF THE SAME CHAT. Holding dead bytes keeps the cache cheap and
# disk-flushable later; we never stash a paused/executing REPL. State lives in
# host memory for the OWUI process lifetime (lost on restart). A falsy chat_id,
# or persistence disabled via valve, yields a fresh stateless REPL each call.

_REPLS: "OrderedDict[str, bytes]" = OrderedDict()  # chat_id -> dump() blob (LRU)


def _get_repl(chat_id: str, persist: bool):
    """Return a MontyRepl for chat_id: rehydrated from the stored blob if one
    exists and persistence is on, else fresh. A falsy chat_id or persist=False
    yields a fresh REPL (stateless)."""
    if persist and chat_id:
        blob = _REPLS.get(chat_id)
        if blob is not None:
            _REPLS.move_to_end(chat_id)  # mark most-recently-used
            return pydantic_monty.MontyRepl.load(blob)
    return pydantic_monty.MontyRepl(script_name="code_mode.py", type_check=False)


def _save_repl(chat_id: str, repl, persist: bool, cache_size: int) -> None:
    """Persist a REPL's state back to the LRU cache as bytes, evicting the
    least-recently-used entries past the cap. No-op when persistence is off or
    chat_id is falsy. If the snippet left the REPL paused (raised mid-execution),
    dump() fails; drop the chat so its next call starts fresh."""
    if not (persist and chat_id):
        return
    try:
        _REPLS[chat_id] = repl.dump()
    except RuntimeError:
        _REPLS.pop(chat_id, None)
        return
    _REPLS.move_to_end(chat_id)
    cap = cache_size if cache_size and cache_size > 0 else 1
    while len(_REPLS) > cap:
        _REPLS.popitem(last=False)


_SCALARS = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
}


def _render_type(schema: dict) -> str:
    """Render a JSON-schema fragment as a Python type-annotation string.

    Faithful to the constructs Open WebUI's spec generator emits (verified
    against the jig fixture): enum -> Literal, anyOf -> Union/Optional,
    array+items -> list[...], array+prefixItems -> tuple[...],
    object+additionalProperties -> dict[str, ...], nested object -> dict.
    Recurses on items/prefixItems/anyOf. Falls back to Any when nothing is
    pinned down.
    """
    if not isinstance(schema, dict):
        return "Any"

    # enum: literal set of allowed values (covers Literal[...] and Enum subclasses)
    enum = schema.get("enum")
    if enum:
        return f"Literal[{', '.join(repr(v) for v in enum)}]"

    # anyOf: Union; a lone non-null arm reads as that arm, null arm -> Optional
    any_of = schema.get("anyOf")
    if any_of:
        arms = [a for a in any_of if a.get("type") != "null"]
        has_null = any(a.get("type") == "null" for a in any_of)
        rendered = [_render_type(a) for a in arms] or ["Any"]
        inner = rendered[0] if len(rendered) == 1 else f"Union[{', '.join(rendered)}]"
        return f"Optional[{inner}]" if has_null else inner

    jtype = schema.get("type")

    if jtype == "array":
        if "prefixItems" in schema:  # fixed-length heterogeneous tuple
            elems = [_render_type(e) for e in schema["prefixItems"]]
            return f"tuple[{', '.join(elems)}]"
        items = schema.get("items")
        if isinstance(items, dict):
            return f"list[{_render_type(items)}]"
        return "list"

    if jtype == "object" or "properties" in schema:
        add = schema.get("additionalProperties")
        if isinstance(add, dict):
            return f"dict[str, {_render_type(add)}]"
        return "dict"

    return _SCALARS.get(jtype, "Any")


def _render_signature(name: str, spec: dict) -> str:
    """Render one OpenAI function spec as a typed Python def + docstring."""
    fn = spec.get("function", spec)  # specs may or may not be {type, function}
    params = fn.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    required = set(params.get("required", []) or [])

    args = []
    for pname, pinfo in props.items():
        ann = _render_type(pinfo)
        if "default" in pinfo:
            # Faithful default straight from the schema (e.g. 1.0, "hello", True).
            args.append(f"{pname}: {ann} = {pinfo['default']!r}")
        elif pname in required:
            args.append(f"{pname}: {ann}")
        else:
            # Optional with no explicit default: present as Optional[...] = None.
            opt = ann if ann.startswith("Optional[") else f"Optional[{ann}]"
            args.append(f"{pname}: {opt} = None")
    sig = f"def {name}({', '.join(args)}) -> Any: ..."

    doc_lines = []
    desc = fn.get("description", "").strip()
    if desc:
        doc_lines.append(desc)
    for pname, pinfo in props.items():
        pdesc = (pinfo.get("description") or "").strip()
        if pdesc:
            doc_lines.append(f"  {pname}: {pdesc}")
    doc = ("\n    " + "\n    ".join(doc_lines)) if doc_lines else ""

    if doc:
        return f'{sig[:-3]}\n    """{doc}\n    """\n    ...'
    return sig


def _build_system_prompt(registry: dict, persist: bool) -> str:
    """Generate the typed-Python API documentation for the wrapped tools."""
    sigs = []
    for name, entry in registry.items():
        sigs.append(_render_signature(name, entry.get("spec", {})))
    api = "\n\n".join(sigs)

    persistence_note = (
        "PERSISTENT STATE: run_python is a REPL, not a fresh sandbox each call. "
        "Variables and functions you define at the top level persist into your "
        "next run_python call in this same conversation. You can build up state "
        "across turns (e.g. accumulate results in a list, define a helper once "
        "and reuse it later). Re-defining or reassigning is fine.\n\n"
        if persist
        else "Each run_python call runs in a fresh sandbox; state does not "
        "persist between calls.\n\n"
    )

    return (
        "You are in CODE MODE. You do not call the wrapped functions directly. "
        "Instead, you have ONE tool: `run_python(code: str)`. Write a short "
        "Python 3 program (string) and pass it as `code`. Inside that program "
        "the following functions are pre-defined and may be called freely "
        "(including in loops, comprehensions, and conditionals). They are "
        "ordinary synchronous-looking calls; the runtime dispatches them to the "
        "real tools. Use `print(...)` for anything you want surfaced back.\n\n"
        "Available wrapped API:\n\n"
        "```python\n"
        "from typing import Any, Optional, Union, Literal\n\n"
        f"{api}\n"
        "```\n\n"
        "Prefer doing as much aggregation/filtering inside the program as "
        "possible so you make a single run_python call rather than many tool "
        "round-trips.\n\n"
        f"{persistence_note}"
        "IMPORTANT: run_python returns a JSON object with the program's actual "
        "stdout and result (or an error). NEVER invent or guess tool results. "
        "Report only what run_python actually returned. If it returns an error "
        "or an empty result, say so plainly rather than fabricating an outcome."
    )


# The single tool the model is allowed to see/call.
RUN_PYTHON_SPEC = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a Python program that may call the pre-defined wrapped "
            "tool functions (see system prompt). Returns the program's stdout."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python 3 source code to execute.",
                }
            },
            "required": ["code"],
        },
    },
}


def _add_or_update_system_message(content: str, messages: list) -> list:
    """Prepend/merge a system message into the OpenAI-style messages list."""
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = content + "\n\n" + messages[0].get("content", "")
    else:
        messages.insert(0, {"role": "system", "content": content})
    return messages


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Filter execution priority (lower runs first)."
        )
        persist: bool = Field(
            default=True,
            description=(
                "Persist the Python REPL state per chat, so variables and "
                "functions the model defines survive across run_python calls "
                "within a conversation. Disable for fully stateless execution."
            ),
        )
        cache_size: int = Field(
            default=128,
            ge=1,
            description=(
                "Maximum number of conversations to keep REPL state for "
                "(LRU eviction). Must be at least 1. Only relevant when persist "
                "is enabled. State is held in process memory and lost on restart."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()
        # toggle=True => per-chat switch in the OWUI integrations menu. Works
        # whether the function is installed globally (opt-in for any chat) or
        # enabled/defaulted on specific models. Do NOT hardcode global here;
        # that is an admin install choice (see README).
        self.toggle = True
        # A distinctive code-brackets icon for the toggle.
        self.icon = (
            "data:image/svg+xml;base64,"
            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
            "NCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0i"
            "MiI+PHBvbHlsaW5lIHBvaW50cz0iMTYgMTggMjIgMTIgMTYgNiIvPjxwb2x5bGluZSBwb2lu"
            "dHM9IjggNiAyIDEyIDggMTgiLz48L3N2Zz4="
        )

    async def inlet(
        self,
        body: dict,
        __request__=None,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __event_emitter__=None,
        __event_call__=None,
        __oauth_token__=None,
        __chat_id__=None,
        __message_id__=None,
    ) -> dict:
        # 1) Discover which toolkits the model has enabled. If none, this filter
        #    has nothing to wrap and passes the request through untouched.
        tool_ids = body.get("tool_ids") or (__metadata__ or {}).get("tool_ids") or []
        if not tool_ids:
            return body

        # 2) Resolve the real callables. get_tools wants a real UserModel, not
        #    the plain __user__ dict.
        user_model = None
        if __user__ and __user__.get("id"):
            user_model = await Users.get_user_by_id(__user__["id"])
        if user_model is None:
            log.warning(f"{MARK} could not resolve UserModel; passing through")
            return body

        # Forward the MAXIMAL real extra_params to get_tools. OWUI's
        # get_async_tool_function_and_apply_extra_params (utils/tools.py)
        # introspects each tool function's signature and binds ONLY the dunders
        # that function actually declares, then partial-binds them so the
        # model-facing signature keeps only the real args. So a tool that wants
        # __event_emitter__ gets the real one; a pure tool gets none. We supply
        # everything real and let OWUI match. These mirror the canonical
        # native-tool extra_params, reconstructed from what our inlet receives.
        extra_params = {
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__user__": __user__,
            "__metadata__": __metadata__,
            "__oauth_token__": __oauth_token__,
            "__request__": __request__,
            "__model__": __model__,
            "__chat_id__": __chat_id__ or (__metadata__ or {}).get("chat_id"),
            "__message_id__": __message_id__
            or (__metadata__ or {}).get("message_id"),
        }

        registry = await get_tools(__request__, tool_ids, user_model, extra_params)
        if not registry:
            log.warning(f"{MARK} get_tools returned empty; passing through")
            return body

        log.info(f"{MARK} wrapping {len(registry)} tool function(s) behind run_python")

        persist = bool(self.valves.persist)
        cache_size = int(self.valves.cache_size)

        # 3) Suppress OWUI's own tool resolution by emptying tool_ids in BOTH
        #    slots (body and metadata). Resolution is gated on `if tool_ids:`;
        #    an empty tools_dict means OWUI never overwrites our seeded
        #    metadata['tools'] nor merges body['tools']. We own both slots.
        body.pop("tool_ids", None)
        if __metadata__ is not None:
            __metadata__["tool_ids"] = []
            # 4a) Seed the EXECUTION registry. The native loop reads
            #     metadata['tools'] and awaits tool['callable']. A plain async
            #     def passes through get_updated_tool_function untouched.
            tools_slot = __metadata__.setdefault("tools", {})
            # Per-chat persistence key. Prefer the explicit __chat_id__ dunder,
            # fall back to metadata; falsy => stateless REPL in the runner.
            _chat_id = extra_params.get("__chat_id__") or ""

            async def run_python(code: str) -> str:
                # Build the external-function table Monty dispatches to: name ->
                # the real OWUI async callable resolved in inlet. That callable
                # already has its needed dunders (__event_emitter__, __user__,
                # etc.) partial-bound by get_tools, so a tool that emits statuses
                # WILL emit them to this chat. We dispatch only the model-facing
                # args from the program; the dunders are frozen.
                callables = {
                    name: entry["callable"] for name, entry in registry.items()
                }

                # Capture print(...) output from inside the sandbox, keeping
                # stdout and stderr separate so we report each only when
                # non-empty.
                stdout_parts: list[str] = []
                stderr_parts: list[str] = []

                def _print_callback(stream: str, text: str) -> None:
                    (stderr_parts if stream == "stderr" else stdout_parts).append(text)

                # Drive Monty's sync start()/resume() loop ourselves. We are
                # already inside OWUI's running event loop and run_python is
                # async, so when Monty pauses on a function call we `await` the
                # real (async) OWUI callable in THIS loop and resume with its
                # value. This keeps the model's ergonomics synchronous (it
                # writes `flip("heads")`, not `await flip(...)`) while still
                # awaiting genuinely-async tools natively.
                #
                # PERSISTENCE: rehydrate this chat's MontyRepl from its stored
                # blob, feed_start() the snippet, and dump the new state back
                # after a clean run. Globals/functions bound in a previous
                # run_python call of this chat are therefore in scope here.
                try:
                    repl = _get_repl(_chat_id, persist)
                    progress = repl.feed_start(code, print_callback=_print_callback)

                    # Monty pauses on two kinds of snapshot we care about:
                    #   FunctionSnapshot   - a call to a host function; fulfill
                    #                        it by awaiting the real OWUI callable.
                    #   NameLookupSnapshot - a reference to a name Monty cannot
                    #                        resolve. We have none to offer, so
                    #                        we resume undefined, surfacing an
                    #                        ordinary NameError to the model.
                    while not isinstance(progress, pydantic_monty.MontyComplete):
                        if isinstance(progress, pydantic_monty.FunctionSnapshot):
                            name = progress.function_name
                            args = progress.args or ()
                            kwargs = progress.kwargs or {}
                            if name not in callables:
                                progress = progress.resume(
                                    {
                                        "exception": NameError(
                                            f"function {name!r} is not available"
                                        )
                                    }
                                )
                                continue
                            try:
                                ret = await callables[name](*args, **kwargs)
                                progress = progress.resume({"return_value": ret})
                            except Exception as call_err:
                                # Surface the tool's own failure back into the
                                # sandbox so the program can see/raise it.
                                progress = progress.resume({"exception": call_err})
                        elif isinstance(progress, pydantic_monty.NameLookupSnapshot):
                            # NameLookupSnapshot.resume has a different API than
                            # FunctionSnapshot.resume. resume(undefined=True)
                            # leaves the name unresolved so Monty raises its own
                            # NameError, which the outer except returns to the
                            # model as a clean error.
                            progress = progress.resume(undefined=True)
                        else:
                            # Any other snapshot (e.g. FutureSnapshot) is out of
                            # scope; report rather than guess.
                            raise RuntimeError(
                                f"unsupported Monty snapshot in runner: "
                                f"{type(progress).__name__}"
                            )

                    result_value = progress.output
                    # Persist the chat's new namespace/heap for the next call.
                    _save_repl(_chat_id, repl, persist, cache_size)
                except Exception as e:  # MontySyntaxError/MontyRuntimeError/etc.
                    log.warning(
                        f"{MARK} program execution failed: {type(e).__name__}"
                    )
                    err_result = {
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                    if stdout_parts:
                        err_result["stdout"] = "".join(stdout_parts)
                    if stderr_parts:
                        err_result["stderr"] = "".join(stderr_parts)
                    return json.dumps(err_result)

                stdout = "".join(stdout_parts)
                stderr = "".join(stderr_parts)
                # Monty returns the value of the last expression. Render it to a
                # string best-effort; non-serializable values fall back to repr.
                try:
                    json.dumps(result_value)
                    result_repr = result_value
                except (TypeError, ValueError):
                    result_repr = repr(result_value)

                ok_result = {"status": "ok", "result": result_repr}
                if stdout:
                    ok_result["stdout"] = stdout
                if stderr:
                    ok_result["stderr"] = stderr
                return json.dumps(ok_result)

            entry = {
                "callable": run_python,
                "spec": RUN_PYTHON_SPEC["function"],
                "type": "function",
            }
            tools_slot["run_python"] = entry
            # Belt-and-suspenders: also seed via body['metadata'] in case the
            # inlet's __metadata__ reference diverges from form_data['metadata']
            # (which is what flows into the execution loop).
            body_meta = body.setdefault("metadata", {})
            body_meta_tools = body_meta.setdefault("tools", {})
            body_meta_tools["run_python"] = entry

        # 4b) Set the MODEL-FACING tool list directly. Because tools_dict stays
        #     empty, OWUI's native build never runs to overwrite this.
        body["tools"] = [RUN_PYTHON_SPEC]

        # 5) Inject the typed-Python API documentation as a system message.
        sys_prompt = _build_system_prompt(registry, persist)
        body["messages"] = _add_or_update_system_message(
            sys_prompt, body.get("messages", [])
        )

        return body

    async def outlet(self, body: dict, **kwargs) -> dict:
        return body
