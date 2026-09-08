from pathlib import Path

import typer

from .ingestion.scanner import scan_path

app = typer.Typer()


# 如果不写以下代码段,typer会简化为单命令模式,调用时不再需要function name
@app.callback()
def main():
    """面经 AI 助手。"""


@app.command()
def scan(path: str):
    """扫描目录下的所有md文件"""
    try:
        result = scan_path(Path(path))
        typer.echo("name\tsize\tcreate_time\tmodify_time\tcontent")
        for item in result:
            typer.echo(
                f"{item.name}\t{item.size:,}\t{item.create_time.strftime('%Y-%m-%d %H:%M:%S')}\t{item.modify_time.strftime('%Y-%m-%d %H:%M:%S')}\t{item.content[:20]!r}"
            )

        typer.echo(f"total={len(result)}")
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except Exception as e:
        typer.echo(e)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
