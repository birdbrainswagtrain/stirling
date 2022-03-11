# Skitter

> Experimental JIT for a primitive subset of Rust

## Features

### Types
- Integers: Literals, arithmetic, comparisons, `as` casts between various widths.
- Bools: Literals

### Control Flow
- blocks, can yield values
- if-else, can yield values
- while loops (no break or continue)
- simple function calls (no method invocation or generics)

### Not Yet Implemented
- i128 / u128 constants may not work.

### Not Supported
- The `mut` keyword is ignored. All values are assumed to be mutable.
