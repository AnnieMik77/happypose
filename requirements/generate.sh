#!/usr/bin/env bash

set -euxo pipefail

# https://stackoverflow.com/a/246128/1368502
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "${SCRIPT_DIR}/.."

# No hashes because:
# ERROR: The editable requirement file:///home/runner/work/happypose/happypose (from -r requirements/cpu.txt (line 3)) cannot be installed when requiring hashes, because there is no single file to hash.

uv export --no-hashes > "${SCRIPT_DIR}/base.txt"
uv export --no-hashes --extra cpu > "${SCRIPT_DIR}/cpu.txt"
uv export --no-hashes --extra cu124 > "${SCRIPT_DIR}/cu124.txt"
uv export --no-hashes --extra pypi > "${SCRIPT_DIR}/pypi.txt"

# TODO: uv should take care of that
echo "--extra-index-url https://download.pytorch.org/whl/cpu" >> "${SCRIPT_DIR}/cpu.txt"
echo "--extra-index-url https://download.pytorch.org/whl/cu124" >> "${SCRIPT_DIR}/cu124.txt"
