# Stirling

> Experimental interpreter for a subset of Rust
## Features
- Simple operations on primitive types: Integers, Floats, Booleans, etc.
- Most control flow: blocks, if, while, loop, break, continue
- Simple function calls (no methods or generics)
- Some compound types: references, pointers, and tuples

### Not Yet Implemented
- Patterns (if-let, match, etc)
- Anything related to namespaces / modules / name resolution is probably severely busted.

### Not Supported
The compiler mostly assumes you know what you're doing. It does not check for unsafety.
- The `mut` keyword is ignored. All values are assumed to be mutable. References track mutability but only for the sake of the type system.
- There is no ownership, lifetimes, or borrow checker.
