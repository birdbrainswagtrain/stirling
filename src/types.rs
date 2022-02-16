
use std::{rc::Rc, cell::RefCell};

use syn::BinOp;

use crate::{hir_expr::Expr, hir_items::{Scope, try_path_to_name}};




pub struct TypeRegistry{}

#[derive(Debug)]
pub struct Signature{
    pub inputs: Vec<Type>,
    pub output: Type
}

impl Signature{
    pub fn from_syn(syn_sig: &syn::Signature, scope: &Scope) -> Signature {
        let mut inputs = Vec::new();
        for arg in &syn_sig.inputs {
            match arg {
                syn::FnArg::Receiver(_) => panic!("recv"),
                syn::FnArg::Typed(x) => {
                    inputs.push(Type::from_syn(&x.ty, scope));
                }
            };
        }
        let output = match &syn_sig.output {
            syn::ReturnType::Default => panic!("default return type"),
            syn::ReturnType::Type(_,syn_ty) => {
                Type::from_syn(&syn_ty, scope)
            }
        };
        Signature{inputs, output}
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Type{
    Unknown,
    IntUnknown,
    Int(TypeInt)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TypeInt{
    I32
}

impl Type {
    pub fn from_syn(syn_ty: &syn::Type, scope: &Scope) -> Type {
        match syn_ty {
            syn::Type::Path(path) => {
                if let Some(name) = try_path_to_name(&path.path) {
                    match name.as_str() {
                        "i32" => Type::Int(TypeInt::I32),
                        _ => panic!("type from name {}",name)
                    }
                } else {
                    panic!("complex path to type")
                }
            }
            _ => panic!("todo convert type! {:?}",syn_ty)
        }
    }

    pub fn is_unknown(&self) -> bool {
        match self {
            Type::Unknown | Type::IntUnknown => true,
            _ => false
        }
    }

    pub fn is_numeric_primitive(&self) -> bool {
        match self {
            Type::IntUnknown | Type::Int(_) => true,
            _ => false
        }
    }

    pub fn more_specific_than(&self, other: Type) -> bool {
        if *self == other {
            false
        } else {
            if *self == Type::Unknown {
                false
            } else {
                true
            }
        }
    }
}
