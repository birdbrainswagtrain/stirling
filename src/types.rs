
use crate::hir_items::{Scope, try_path_to_name};

const PTR_WIDTH: usize = 8;


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
    Int(TypeInt),
    Void,
    Never
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TypeInt{
    ISize,
    I128,
    I64,
    I32,
    I16,
    I8,

    USize,
    U128,
    U64,
    U32,
    U16,
    U8
}

impl Type {
    pub fn from_syn(syn_ty: &syn::Type, scope: &Scope) -> Type {
        match syn_ty {
            syn::Type::Path(path) => {
                if let Some(name) = try_path_to_name(&path.path) {
                    Self::from_str(&name).expect("failed to parse type")
                } else {
                    panic!("complex path to type")
                }
            }
            _ => panic!("todo convert type! {:?}",syn_ty)
        }
    }

    pub fn from_str(name: &str) -> Option<Type> {
        match name {

            "isize" => Some( Type::Int(TypeInt::ISize) ),
            "i128" => Some( Type::Int(TypeInt::I128) ),
            "i64" => Some( Type::Int(TypeInt::I64) ),
            "i32" => Some( Type::Int(TypeInt::I32) ),
            "i16" => Some( Type::Int(TypeInt::I16) ),
            "i8" => Some( Type::Int(TypeInt::I8) ),

            "usize" => Some( Type::Int(TypeInt::USize) ),
            "u128" => Some( Type::Int(TypeInt::U128) ),
            "u64" => Some( Type::Int(TypeInt::U64) ),
            "u32" => Some( Type::Int(TypeInt::U32) ),
            "u16" => Some( Type::Int(TypeInt::U16) ),
            "u8" => Some( Type::Int(TypeInt::U8) ),

            _ => None
        }
    }

    pub fn is_unknown(&self) -> bool {
        match self {
            Type::Unknown | Type::IntUnknown => true,
            _ => false
        }
    }

    pub fn is_number(&self) -> bool {
        match self {
            Type::IntUnknown | Type::Int(_) => true,
            _ => false
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Type::IntUnknown | Type::Int(_) => true,
            _ => false
        }
    }

    pub fn is_signed(&self) -> bool {
        if let Type::Int(ti) = self {
            match ti {
                TypeInt::ISize | TypeInt::I128 | TypeInt::I64 | TypeInt::I32 | TypeInt::I16 | TypeInt::I8 => true,
                TypeInt::USize | TypeInt::U128 | TypeInt::U64 | TypeInt::U32 | TypeInt::U16 | TypeInt::U8 => false
            }
        } else {
            panic!("can't check signed-ness of {:?}",self)
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Type::Int(TypeInt::I64) | Type::Int(TypeInt::U64) => 8,
            Type::Int(TypeInt::I32) | Type::Int(TypeInt::U32) => 4,
            Type::Int(TypeInt::I16) | Type::Int(TypeInt::U16) => 2,
            Type::Int(TypeInt::I8)  | Type::Int(TypeInt::U8)  => 1,
            _ => panic!("cannot size {:?}",self)
        }
    }

    pub fn can_upgrade_to(self, other: Type) -> bool {
        if self == other {
            false
        } else {
            match (self,other) {
                (Type::Unknown,_) => true,
                (Type::IntUnknown,Type::Int(_)) => true,

                (_,Type::Unknown) => false,
                (Type::Int(_),Type::IntUnknown) => false,
                _ => panic!("type error, can not unify types {:?} and {:?}",self,other)
            }
        }
    }
}
