#!/usr/bin/env sh
# Tiny wait-for replacement (no netcat required)
# Usage: wait-for.sh host:port -- command args...
HOSTPORT="$1"
shift
HOST=$(echo "$HOSTPORT" | cut -d: -f1)
PORT=$(echo "$HOSTPORT" | cut -d: -f2)

echo "Waiting for $HOST:$PORT..."
while ! (echo > /dev/tcp/$HOST/$PORT) 2>/dev/null; do
  sleep 1
done
echo "Up!"

exec "$@"
