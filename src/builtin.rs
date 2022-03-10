
use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::types::{Signature, Type, TypeInt};

pub static BUILTINS: Lazy<HashMap<&'static str,(usize,Signature)>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("print_i32",(_builtin::print_i32 as _, Signature::new(vec!(Type::Int(TypeInt::I32)), Type::Void)));
    m
});

mod _builtin {
    include!("../test/_skitter_builtin.rs");
}
