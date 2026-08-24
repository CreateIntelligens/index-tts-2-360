#!/usr/bin/env bash
# Shared connection settings for the Taipei-1 scripts, sourced not run.
#
# The host names, account and key live in the environment rather than in the
# scripts. They are infrastructure this project was lent for a fixed window,
# not something to carry in a public history, and the next person to run these
# will have different ones.
#
# Put them in ~/.env.taipei1 (or point TP1_ENV elsewhere):
#
#   TP1_HOST=user@login-node
#   TP1_JUMP=user@jump-host
#   TP1_KEY=$HOME/.ssh/some_ed25519    # default ~/.ssh/id_ed25519
#   TP1_PORT=2222                      # default 2222
#
# Every hop goes through the jump host, so ProxyCommand is not optional here.

TP1_ENV=${TP1_ENV:-$HOME/.env.taipei1}
# shellcheck source=/dev/null
[ -f "$TP1_ENV" ] && . "$TP1_ENV"

TP1_HOST=${TP1_HOST:?請設定 TP1_HOST=user@login-node（見 tools/taipei1_env.sh）}
TP1_JUMP=${TP1_JUMP:?請設定 TP1_JUMP=user@jump-host（見 tools/taipei1_env.sh）}
TP1_KEY=${TP1_KEY:-$HOME/.ssh/id_ed25519}
TP1_PORT=${TP1_PORT:-2222}

# Resolved on the host, because the container that does the NAS writes has none
# of these variables and sees the key at /keys/<name> instead.
KEYDIR=$(dirname "$TP1_KEY")
KEYNAME=$(basename "$TP1_KEY")

SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=30"
# No known_hosts inside a throwaway container, and the key is pinned by -i.
SSH_OPTS_CONT="$SSH_OPTS -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# For rsync -e / scp run on this machine.
SSH_CMD_HOST="ssh $SSH_OPTS -i $TP1_KEY -p $TP1_PORT \
-o \"ProxyCommand=ssh $SSH_OPTS -i $TP1_KEY -W %h:%p -p $TP1_PORT $TP1_JUMP\""
# Same thing with the container's view of the key.
SSH_CMD_CONT="ssh $SSH_OPTS_CONT -i /keys/$KEYNAME -p $TP1_PORT \
-o \"ProxyCommand=ssh $SSH_OPTS_CONT -i /keys/$KEYNAME -W %h:%p -p $TP1_PORT $TP1_JUMP\""

# Run a command on the login node. Used with redirection, so it must stay a
# plain command that writes the remote stdout to ours.
tp1() {
    ssh $SSH_OPTS -i "$TP1_KEY" -p "$TP1_PORT" \
        -o "ProxyCommand=ssh $SSH_OPTS -i $TP1_KEY -W %h:%p -p $TP1_PORT $TP1_JUMP" \
        "$TP1_HOST" "$@"
}
