use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::hir::types::{FloatType, IntType, Signature, Type};
use crate::jit::jit_compile;

pub static BUILTINS: Lazy<HashMap<&'static str, (usize, Signature)>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(
        "jit_compile",
        (
            jit_compile as _,
            Signature::new(vec![Type::Int(IntType::USize)], Type::Int(IntType::USize)),
        ),
    );
    m.insert(
        "print_i64",
        (
            _builtin::print_i64 as _,
            Signature::new(vec![Type::Int(IntType::I64)], Type::Void),
        ),
    );
    m.insert(
        "print_header",
        (
            _builtin::print_header as _,
            Signature::new(vec![Type::Int(IntType::I64)], Type::Void),
        ),
    );
    m.insert(
        "print_f64",
        (
            _builtin::print_f64 as _,
            Signature::new(vec![Type::Float(FloatType::F64)], Type::Void),
        ),
    );
    m.insert(
        "print_char",
        (
            _builtin::print_char as _,
            Signature::new(vec![Type::Char], Type::Void),
        ),
    );
    m
});

mod _builtin {
    include!("../test/_skitter_builtin.rs");
}
