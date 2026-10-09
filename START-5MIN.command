#!/bin/zsh
TASK_ROOT="${0:A:h}"
exec "$TASK_ROOT/START.command" --test "$@"
