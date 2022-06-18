use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::hir::types::{FloatType, IntType, Signature, Type};

pub static BUILTINS: Lazy<HashMap<&'static str, (usize, Signature)>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(
        "print_int",
        (
            builtin::print_int as _,
            Signature::new(vec![Type::Int(IntType::I128)], Type::Void),
        ),
    );
    m.insert(
        "print_uint",
        (
            builtin::print_uint as _,
            Signature::new(vec![Type::Int(IntType::U128)], Type::Void),
        ),
    );
    m.insert(
        "print_float",
        (
            builtin::print_float as _,
            Signature::new(vec![Type::Float(FloatType::F64)], Type::Void),
        ),
    );
    m.insert(
        "print_bool",
        (
            builtin::print_bool as _,
            Signature::new(vec![Type::Bool], Type::Void),
        ),
    );
    m.insert(
        "print_char",
        (
            builtin::print_char as _,
            Signature::new(vec![Type::Char], Type::Void),
        ),
    );
    /*m.insert(
        "print_i64",
        (
            builtin::print_i64 as _,
            Signature::new(vec![Type::Int(IntType::I64)], Type::Void),
        ),
    );
    m.insert(
        "assert",
        (
            builtin::assert as _,
            Signature::new(vec![Type::Int(IntType::I32), Type::Bool], Type::Void),
        ),
    );
    m.insert(
        "print_f64",
        (
            builtin::print_f64 as _,
            Signature::new(vec![Type::Float(FloatType::F64)], Type::Void),
        ),
    );
    m.insert(
        "print_char",
        (
            builtin::print_char as _,
            Signature::new(vec![Type::Char], Type::Void),
        ),
    );*/
    m
});

mod builtin;
pub use builtin::*;
