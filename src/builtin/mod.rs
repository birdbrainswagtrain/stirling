use std::collections::HashMap;

use crate::hir::types::common::{TypeKind,IntWidth,IntSign,FloatWidth};
use crate::hir::types::global::{Signature,GlobalType};

use once_cell::sync::Lazy;

mod builtin;
pub use builtin::*;

pub static BUILTINS: Lazy<HashMap<&'static str, (usize, Signature)>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(
        "print_int",
        (
            builtin::print_int as _,
            Signature::new(vec![
                GlobalType::simple(TypeKind::Int(Some((IntWidth::Int128,IntSign::Signed))))
            ],
            GlobalType::simple(TypeKind::Tuple))
        ),
    );
    m.insert(
        "print_uint",
        (
            builtin::print_uint as _,
            Signature::new(vec![
                GlobalType::simple(TypeKind::Int(Some((IntWidth::Int128,IntSign::Unsigned))))
            ],
            GlobalType::simple(TypeKind::Tuple))
        ),
    );
    m.insert(
        "print_float",
        (
            builtin::print_float as _,
            Signature::new(vec![
                GlobalType::simple(TypeKind::Float(Some(FloatWidth::Float64)))
            ],
            GlobalType::simple(TypeKind::Tuple))
        ),
    );
    m.insert(
        "print_bool",
        (
            builtin::print_bool as _,
            Signature::new(vec![
                GlobalType::simple(TypeKind::Bool)
            ],
            GlobalType::simple(TypeKind::Tuple))
        ),
    );
    m.insert(
        "print_char",
        (
            builtin::print_char as _,
            Signature::new(vec![
                GlobalType::simple(TypeKind::Char)
            ],
            GlobalType::simple(TypeKind::Tuple))
        ),
    );
    m
});
