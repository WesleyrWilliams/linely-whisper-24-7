#!/usr/bin/env bash
# Render build script

set -o errexit
set -o pipefail
set -o nounset

echo "---- Upgrading pip and installing build tools ----"
pip install --upgrade pip setuptools wheel build

echo "---- Installing dependencies ----"
pip install -r requirements.txt

echo "---- Build step complete ----"

