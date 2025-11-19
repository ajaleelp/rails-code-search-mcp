#!/usr/bin/env python3
"""Terminal REPL that lets Claude call code search MCP tools."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import itertools
from typing import Any, Dict, List

from anthropic import Anthropic
from openai import OpenAI
from sqlalchemy import create_engine

from rails_code_search.config import SearchConfig
from rails_code_search.mcp_server import CodeSearchMCPServer

SYSTEM_PROMPT = """You are a code navigation assistant helping developers understand and triage code issues.

You have access to these tools:
- search_code: Semantic search using natural language queries
- list_features: Show all available features/modules in the codebase
- search_by_feature: Get all code in a specific feature
- search_by_class: Find all methods in a class
- get_file_chunks: Get all code chunks from a file
- get_surrounding_context: Get code around a specific line number
- search_related_code: Find code that references a class/method

Best practices:
1. Start with list_features to understand the codebase structure
2. Use search_code for initial discovery with specific, detailed queries
3. When you find relevant code, use get_file_chunks to see complete files
4. Use search_by_class to explore class structure
5. Use get_surrounding_context to understand code context
6. Use search_related_code to find usages and dependencies

When helping with an issue, be systematic:
- Ask clarifying questions if needed
- List features first to understand scope
- Search semantically with detailed queries
- Explore classes and files thoroughly
- Provide file paths and line numbers in your answers
"""

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")


def build_server(database_url: str, openai_api_key: str) -> CodeSearchMCPServer:
    """Build the code search MCP server."""
    config = SearchConfig(
        database_url=database_url,
        openai_api_key=openai_api_key,
    )
    return CodeSearchMCPServer(config)


def get_tools(server: CodeSearchMCPServer) -> List[Dict[str, Any]]:
    """Get all available MCP tools."""
    return server.list_tools()


def execute_tool(server: CodeSearchMCPServer, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool and return results."""
    try:
        if name == "search_code":
            return server.search_code(args)
        elif name == "list_features":
            return server.list_features(args)
        elif name == "search_by_feature":
            return server.search_by_feature(args)
        elif name == "search_by_class":
            return server.search_by_class(args)
        elif name == "get_file_chunks":
            return server.get_file_chunks(args)
        elif name == "get_surrounding_context":
            return server.get_surrounding_context(args)
        elif name == "search_related_code":
            return server.search_related_code(args)
        else:
            raise ValueError(f"Unknown tool '{name}'")
    except ValueError as exc:
        return {"error": "validation_error", "message": str(exc)}
    except Exception as exc:
        return {"error": "execution_error", "message": str(exc)}


class Spinner:
    """Animated spinner for terminal."""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.stdout.flush()

    def _run(self):
        for frame in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{self.message} {frame}")
            sys.stdout.flush()
            time.sleep(0.1)


def format_tool_summary(name: str, result: Dict[str, Any]) -> str:
    """Format a brief summary of tool execution."""
    if "error" in result:
        return f"[{name}] error: {result['error']} - {result.get('message', '')}"

    if name == "search_code":
        count = result.get("result_count", 0)
        return f"[{name}] found {count} result(s)"
    elif name == "list_features":
        count = result.get("feature_count", 0)
        return f"[{name}] found {count} feature(s)"
    elif name == "search_by_feature":
        count = result.get("chunk_count", 0)
        feature = result.get("feature", "")
        return f"[{name}] found {count} chunk(s) in '{feature}'"
    elif name == "search_by_class":
        count = result.get("method_count", 0)
        class_name = result.get("class_name", "")
        return f"[{name}] found {count} method(s) in '{class_name}'"
    elif name == "get_file_chunks":
        count = result.get("chunk_count", 0)
        return f"[{name}] retrieved {count} chunk(s)"
    elif name == "get_surrounding_context":
        count = result.get("chunk_count", 0)
        return f"[{name}] retrieved {count} context chunk(s)"
    elif name == "search_related_code":
        count = result.get("result_count", 0)
        return f"[{name}] found {count} reference(s)"

    return f"[{name}] completed"


def main() -> None:
    """Run the interactive assistant REPL."""
    required_env = ["CODE_SEARCH_DATABASE_URL", "CODE_SEARCH_OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [var for var in required_env if not os.environ.get(var)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    database_url = os.environ["CODE_SEARCH_DATABASE_URL"]
    openai_api_key = os.environ["CODE_SEARCH_OPENAI_API_KEY"]

    server = build_server(database_url, openai_api_key)
    tools = get_tools(server)

    client = Anthropic()
    history: List[Dict[str, Any]] = []

    print("=" * 60)
    print("Code Search Assistant - Ready to help you navigate code!")
    print("=" * 60)
    print("\nTips:")
    print("  - Ask about specific features or issues")
    print("  - I'll use multiple tools to find relevant code")
    print("  - Type 'quit' or 'exit' to leave")
    print("\nExample queries:")
    print("  'Show me all features in this codebase'")
    print("  'Find code related to pulse age calculation'")
    print("  'What classes are in the vitamin_d feature?'")
    print()

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        history.append({"role": "user", "content": prompt})

        # Agentic loop: keep calling tools until Claude is done
        while True:
            with Spinner():
                response = client.messages.create(
                    model=DEFAULT_MODEL,
                    system=SYSTEM_PROMPT,
                    messages=history,
                    tools=tools,
                    max_tokens=4096,
                    temperature=0.1,
                )

            assistant_blocks: List[Dict[str, Any]] = []
            tool_results: List[tuple[str, str]] = []

            for block in response.content:
                if block.type == "text":
                    print(f"\nClaude> {block.text}")
                    assistant_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                    result = execute_tool(server, block.name, block.input)
                    serialized = json.dumps(result, indent=2, default=str)
                    print(format_tool_summary(block.name, result))
                    tool_results.append((block.id, serialized))

            if assistant_blocks:
                history.append({"role": "assistant", "content": assistant_blocks})

            # If no tool calls, Claude is done
            if not tool_results:
                break

            # Add tool results to history for next iteration
            for tool_use_id, serialized in tool_results:
                history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": serialized,
                                    }
                                ],
                            }
                        ],
                    }
                )

        print()  # Add newline after complete interaction


if __name__ == "__main__":
    main()
