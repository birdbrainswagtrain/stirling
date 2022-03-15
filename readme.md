# Skitter

> Experimental JIT for a primitive subset of Rust

## Features

### Types
- Integers (excluding i128 / u128): Literals, arithmetic, bitwise, comparisons, `as` casts between various widths.
- Bools: Literals, logic, bitwise (IE non-lazy logic), equality. **(TODO as casts)**
- Floats: Literals, arithmetic, comparisons.
- Chars: Literals, comparisons.

### Control Flow
- blocks, can yield values
- if-then and if-else, can yield values
- while loops
- break and continue for supported loop kinds
- simple function calls (no method invocation or generics)

### Not Yet Implemented
- i128 / u128 do not work
- Patterns, match guards, if-let, etc
- Anything related to namespaces / modules / name resolution is probably severely busted.

### Not Supported
- The `mut` keyword is ignored. All values and references are assumed to be mutable.
