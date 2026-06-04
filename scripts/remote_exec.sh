#!/bin/bash
# remote_exec.sh — Non-blocking remote SSH via tmux with real-time streaming
# Usage: ./remote_exec.sh <host> <command> [timeout=300]
# Output streams to stdout in real-time for live visibility.

HOST="${1:?Usage: $0 <host> <command> [timeout]}"
CMD="${2:?Usage: $0 <host> <command> [timeout]}"
MAX_WAIT="${3:-300}"

PID_TAG="$$"
OUTFILE="/tmp/.ssh_out_${PID_TAG}.txt"
SESSION="ssh_${PID_TAG}"
MARK="@@DONE@@"

cleanup() { tmux kill-session -t "$SESSION" 2>/dev/null; rm -f "$OUTFILE"; }
trap cleanup EXIT

: > "$OUTFILE"

tmux new-session -d -s "$SESSION" \
  "ssh -t -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
   $HOST '${CMD}; echo exit_flag=\$?; echo $MARK'"
tmux pipe-pane -t "$SESSION" "cat >> $OUTFILE"

# Track how many bytes we've already printed for streaming
printed_bytes=0
elapsed=0

while [ $elapsed -lt $MAX_WAIT ]; do
  # Stream new content to stdout in real-time
  if [ -f "$OUTFILE" ]; then
    current_size=$(wc -c < "$OUTFILE" 2>/dev/null | tr -d ' ')
    current_size="${current_size:-0}"
    if [ "$current_size" -gt "$printed_bytes" ]; then
      new_content=$(tail -c +"$((printed_bytes + 1))" "$OUTFILE" 2>/dev/null | col -b)
      # Filter out control markers before printing
      filtered=$(echo "$new_content" | grep -v 'exit_flag=' | grep -vF "$MARK" \
        | grep -v 'Connection to .* closed')
      if [ -n "$filtered" ]; then
        echo "$filtered"
      fi
      printed_bytes=$current_size
    fi
  fi

  # Check if command has finished
  if tail -5 "$OUTFILE" 2>/dev/null | grep -qF "$MARK"; then
    # Flush any remaining unprinted content
    current_size=$(wc -c < "$OUTFILE" 2>/dev/null | tr -d ' ')
    current_size="${current_size:-0}"
    if [ "$current_size" -gt "$printed_bytes" ]; then
      tail -c +"$((printed_bytes + 1))" "$OUTFILE" 2>/dev/null | col -b \
        | grep -v 'exit_flag=' | grep -vF "$MARK" \
        | grep -v 'Connection to .* closed' | sed '/^[[:space:]]*$/d'
    fi
    RC=$(tail -5 "$OUTFILE" | grep 'exit_flag=' | tail -1 | sed 's/.*exit_flag=//' | tr -dc '0-9')
    exit "${RC:-1}"
  fi

  sleep 1
  elapsed=$((elapsed + 1))
done

echo "[TIMEOUT] ${MAX_WAIT}s exceeded." >&2
exit 124
