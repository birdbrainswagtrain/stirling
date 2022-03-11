# Skitter

> Experimental JIT for a primitive subset of Rust

## Features

### Types
- Integers: Literals, arithmetic, bitwise, comparisons, `as` casts between various widths. TODO shifts, equality
- Bools: Literals, "bitwise", TODO ordinal, equality, logic

### Control Flow
- blocks, can yield values
- if-then and if-else, can yield values
- while loops (no break or continue)
- simple function calls (no method invocation or generics)

### Not Yet Implemented
- i128 / u128 constants may not work.

### Not Supported
- The `mut` keyword is ignored. All values are assumed to be mutable.
