# Skitter

> Experimental JIT for a primitive subset of Rust

## Features

### Types
- Bools: Literals, logic, bitwise (IE non-lazy logic), equality. **(TODO as casts)**
- Chars: Literals, comparisons.
- Integers: Literals, arithmetic, bitwise, comparisons, `as` casts between various widths.
- Floats: TODO NEXT

### Control Flow
- blocks, can yield values
- if-then and if-else, can yield values
- while loops (no break or continue)
- simple function calls (no method invocation or generics)

### Not Yet Implemented
- i128 / u128 constants may not work.
- Patterns, match guards, if-let, etc
- Anything related to namespaces / modules / name resolution is probably severely busted.

### Not Supported
- The `mut` keyword is ignored. All values and references are assumed to be mutable.
