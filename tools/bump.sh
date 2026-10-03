#!/bin/sh
# Move the tandem-c pin to the latest upstream main.
git -C "$(dirname "$0")/.." submodule update --remote external/tandem-c
