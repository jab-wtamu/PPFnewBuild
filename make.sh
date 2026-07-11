#!/bin/bash

set -e

module purge
module load foss/2023b CMake/3.27.6 VTK/9.3.0 WebProxy

PRISMS_DIR="/scratch/user/u.jb337251/PRISMS-PF3_PKG"

export PRISMS_PF_DIR="$PRISMS_DIR/PF300"
export PATH="$PRISMS_PF_DIR/bin:$PATH"

echo "Running in: $(pwd)"
echo "PRISMS_DIR=$PRISMS_DIR"
echo "PRISMS_PF_DIR=$PRISMS_PF_DIR"

rm -rf build main-release main-debug

cmake -B build -DCMAKE_BUILD_TYPE=DebugRelease
cmake --build build
