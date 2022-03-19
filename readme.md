# Skitter

> Experimental JIT for a primitive subset of Rust

## Features

### Types
- Integers (excluding i128 / u128): Literals, arithmetic, bitwise, comparisons, `as` casts between various widths.
- Bools: Literals, logic, bitwise (IE non-lazy logic), equality. **(TODO as casts)**
- Floats: Literals, arithmetic, comparisons.
- Chars: Literals, comparisons.
- References: Scalar types can be reference and dereferenced.

### Control Flow
- blocks, can yield values
- `if` expressions, can yield values
- `while` loops
- `loop` loops, can yield values with `break`
- `break` and `continue` for supported loop kinds, supports labels
- simple function calls (no method invocation or generics)

### Not Yet Implemented
- i128 / u128 do not work
- Patterns, match guards, if-let, etc
- Anything related to namespaces / modules / name resolution is probably severely busted.

### Not Supported
- The `mut` keyword is ignored. All values are assumed to be mutable. References track mutability but only for the sake of the type system.
