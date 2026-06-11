ROOT="$(pwd -P)"
LOCAL_PYTHONPATH="$ROOT/src:$ROOT"

run_in_terminal () {
  osascript -e 'on run argv
    tell application "Terminal" to do script "cd " & quoted form of item 1 of argv & " && PYTHONPATH=" & quoted form of item 2 of argv & " uv run python " & item 3 of argv
  end run' "$ROOT" "$LOCAL_PYTHONPATH" "$1"
}

run_in_terminal "./examples/camera.py server"
run_in_terminal "./examples/cli.py api"
run_in_terminal "./examples/cli.py client"