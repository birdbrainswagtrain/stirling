#!/bin/bash
diff --color <(./run.sh $1) <(../target/release/stirling $1)
