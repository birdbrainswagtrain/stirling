#!/bin/bash
bin_name=$(mktemp)
rustc $1 -o $bin_name -C overflow-checks=off
$bin_name
rm $bin_name
