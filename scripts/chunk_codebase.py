#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

TARGET_EXTENSIONS = {".rb", ".md", ".yml"}
RUBY_EXTRACTOR = Path(__file__).with_name("ruby_method_extractor.rb")

def iter_files(root: Path):
  if root.is_file():
    if root.suffix in TARGET_EXTENSIONS:
      yield root
    return

  for path in root.rglob("*"):
    if path.suffix in TARGET_EXTENSIONS and path.is_file():
      yield path

def chunk_plain_file(path: Path, max_lines=80):
  lines = path.read_text(encoding="utf-8").splitlines()
  chunk = []
  start_line = 1
  for idx, line in enumerate(lines, start=1):
    chunk.append(line)
    if len(chunk) >= max_lines:
      yield {"path": str(path), "start_line": start_line, "text": "\n".join(chunk)}
      chunk = []
      start_line = idx + 1
  if chunk:
    yield {"path": str(path), "start_line": start_line, "text": "\n".join(chunk)}


def split_method_chunks(path: Path, header: str, body_lines, start_line: int, max_lines: int, class_name=None, method_name=None):
  buffer = []
  buffer_start = start_line
  for idx, line in enumerate(body_lines, start=0):
    buffer.append(line)
    if len(buffer) >= max_lines:
      text = header + "\n".join(buffer)
      yield {
        "path": str(path),
        "start_line": buffer_start,
        "text": text,
        "class_name": class_name,
        "method_name": method_name
      }
      buffer = []
      buffer_start = start_line + idx + 1
  if buffer:
    text = header + "\n".join(buffer)
    yield {
      "path": str(path),
      "start_line": buffer_start,
      "text": text,
      "class_name": class_name,
      "method_name": method_name
    }


def chunk_ruby_file(path: Path, max_lines=80):
  if not RUBY_EXTRACTOR.exists():
    yield from chunk_plain_file(path, max_lines)
    return

  try:
    result = subprocess.run([
      "ruby",
      str(RUBY_EXTRACTOR),
      str(path)
    ], capture_output=True, text=True, check=True)
    entries = json.loads(result.stdout or "[]")
  except (subprocess.CalledProcessError, json.JSONDecodeError):
    yield from chunk_plain_file(path, max_lines)
    return

  if not entries:
    yield from chunk_plain_file(path, max_lines)
    return

  for entry in entries:
    start_line = entry.get("start_line", 1)
    end_line = entry.get("end_line", start_line)
    body = entry.get("text", "")
    lines = body.splitlines()
    class_name = entry.get("class_name") or entry.get("class")
    method_name = entry.get("method_name") or entry.get("name") or entry.get("method")
    header_parts = []
    if class_name:
      header_parts.append(class_name)
    if method_name:
      header_parts.append(method_name)
    header_label = path.name if not header_parts else "#".join(header_parts)
    header = f"# {header_label} ({path}:{start_line}-{end_line})\n"

    if not lines:
      yield {
        "path": str(path),
        "start_line": start_line,
        "text": header,
        "class_name": class_name,
        "method_name": method_name
      }
    else:
      yield from split_method_chunks(path, header, lines, start_line, max_lines, class_name, method_name)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True, help="Path to repo section to chunk")
  parser.add_argument("--max-lines", type=int, default=80)
  parser.add_argument("--output", required=True, help="JSONL file to write chunks")
  args = parser.parse_args()

  root = Path(args.root)
  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with output_path.open("w", encoding="utf-8") as out:
    for file_path in iter_files(root):
      if file_path.suffix == ".rb":
        chunk_iter = chunk_ruby_file(file_path, max_lines=args.max_lines)
      else:
        chunk_iter = chunk_plain_file(file_path, max_lines=args.max_lines)

      for chunk in chunk_iter:
        payload = {
          "path": chunk["path"],
          "start_line": chunk["start_line"],
          "text": chunk["text"],
          "class_name": chunk.get("class_name"),
          "method_name": chunk.get("method_name")
        }
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")

if __name__ == "__main__":
  main()
