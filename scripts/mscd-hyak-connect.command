#!/bin/bash
# Authenticate interactively and provide the shared socket used by the pilot monitor.
set -eu

mscd_socket=/tmp/mscd-hyak-control

if ssh -S "$mscd_socket" -O check klone >/dev/null 2>&1; then
    printf 'The Hyak connection is already available.\n'
    exit 0
fi

printf 'Log in to Hyak below, including the usual MFA prompt.\n'
ssh -M -S "$mscd_socket" \
    -o ControlPersist=12h \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -fN klone

ssh -S "$mscd_socket" -o BatchMode=yes -o ConnectTimeout=10 klone hostname
printf '\nHyak is connected. The pilot monitor can use this connection.\n'
