"""以 HTTP 传输启动 Brain MCP 服务。"""

from brain.serve.server import mcp


def main() -> None:
    mcp.run(transport="http", host="0.0.0.0", port=2418, path="/mcp")


if __name__ == "__main__":
    main()
