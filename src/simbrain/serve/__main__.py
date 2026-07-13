"""以标准输入输出传输启动 SimBrain MCP 服务。"""

from simbrain.serve.server import mcp


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
