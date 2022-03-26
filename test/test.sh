#!/bin/bash
diff --color <(./run.sh $1 2> /dev/null) <(../target/release/stirling $1)
