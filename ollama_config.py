import os
import re
import socket
import subprocess
from urllib.parse import urlparse

def _parse_hostport(value: str):
    if not value:
        return None, None
    if "://" not in value:
        value = "http://" + value
    u = urlparse(value)
    print(f"Parsed URL: {u}")
    return (u.hostname, u.port)

def detect_ollama_hostport():
    # 1) Prefer OLLAMA_HOST if set (e.g., "sg013:11437" or "http://sg013:11437")
    host, port = _parse_hostport(os.getenv("OLLAMA_HOST"))
    if port:
        # Normalize unroutable binds to the node's hostname
        if host in (None, "0.0.0.0", "127.0.0.1", "localhost", "::"):
            host = socket.gethostname()
        return host, port

    # 2) Fallback: parse listening sockets for the ollama process (Linux)
    # Works on most HPC/servers that have `ss` or `netstat`
    patterns = [
        (["ss", "-plnt"], re.compile(r"LISTEN.*\s([\[\]a-fA-F0-9\.:]+):(\d+).*ollama", re.I)),
        (["netstat", "-plnt"], re.compile(r"LISTEN\s+\d+\s+\d+\s+([\[\]a-fA-F0-9\.:]+):(\d+).+ollama", re.I)),
    ]
    for cmd, rx in patterns:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        except Exception as e:
            print(f"Error executing {cmd}: {e}")
            continue
        for line in out.splitlines():
            print("LINE:", line)
            if "ollama" not in line.lower():
                continue
            m = rx.search(line)
            if m:
                print(f"Found Ollama listening socket: {line.strip()}")
                h, p = m.group(1).strip("[]"), int(m.group(2))
                if h in ("0.0.0.0", "127.0.0.1", "localhost", "::"):
                    h = socket.gethostname()
                return h, p

    # 3) Last resort: assume default
    return socket.gethostname(), 11434

def ollama_base_url() -> str:
    h, p = detect_ollama_hostport()
    return f"http://apoojar4@{h}:{p}/v1"

if __name__ == "__main__":
    host, port = detect_ollama_hostport()
    print(f"Detected Ollama at {host}:{port}")
    print("Base URL:", ollama_base_url())

    # Example with OpenAI-compatible client:
    # from openai import OpenAI
    # client = OpenAI(base_url=ollama_base_url(), api_key="ollama")
