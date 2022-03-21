use once_cell::sync::Lazy;

use std::{collections::HashSet, sync::RwLock};

use crate::PTR_WIDTH;

use super::item::{try_path_to_name, Scope};

struct TypeRegistry {
    lookup: HashSet<&'static ComplexType>,
}

static TYPE_REGISTRY: Lazy<RwLock<TypeRegistry>> = Lazy::new(|| {
    RwLock::new(TypeRegistry {
        lookup: HashSet::new(),
    })
});

#[derive(Debug)]
pub struct Signature {
    pub inputs: Vec<Type>,
    pub output: Type,
}

impl Signature {
    pub fn new(inputs: Vec<Type>, output: Type) -> Self {
        Self { inputs, output }
    }

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
            syn::ReturnType::Default => Type::Void,
            syn::ReturnType::Type(_, syn_ty) => Type::from_syn(&syn_ty, scope),
        };
        Signature { inputs, output }
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub enum ComplexType {
    Ref(Type, bool),
    Ptr(Type, bool),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Type {
    Unknown,
    IntUnknown,
    Int(IntType),
    FloatUnknown,
    Float(FloatType),
    Bool,
    Char,
    Void,
    Never,
    Complex(&'static ComplexType),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntType {
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
    U8,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatType {
    F64,
    F32,
}

impl Type {
    pub fn from_syn(syn_ty: &syn::Type, scope: &Scope) -> Type {
        match syn_ty {
            syn::Type::Path(path) => {
                if let Some(name) = try_path_to_name(&path.path) {
                    if let Some(ty) = Self::from_str(&name) {
                        return ty;
                    }
                }
            }
            syn::Type::Reference(tr) => {
                let is_mut = tr.mutability.is_some();
                let inner = Type::from_syn(&tr.elem, scope);
                return Type::from_complex(ComplexType::Ref(inner, is_mut));
            }
            syn::Type::Ptr(tp) => {
                let is_mut = tp.mutability.is_some();
                let inner = Type::from_syn(&tp.elem, scope);
                return Type::from_complex(ComplexType::Ptr(inner, is_mut));
            }
            syn::Type::Never(_) => return Type::Never,
            syn::Type::Infer(_) => return Type::Unknown,
            syn::Type::Tuple(syn::TypeTuple { elems, .. }) => {
                if elems.len() == 0 {
                    return Type::Void;
                } else {
                    panic!("todo real tuple types");
                }
            }
            _ => (),
        }
        panic!("failed to convert type {:?}", syn_ty)
    }

    pub fn from_str(name: &str) -> Option<Type> {
        match name {
            "isize" => Some(Type::Int(IntType::ISize)),
            "i128" => Some(Type::Int(IntType::I128)),
            "i64" => Some(Type::Int(IntType::I64)),
            "i32" => Some(Type::Int(IntType::I32)),
            "i16" => Some(Type::Int(IntType::I16)),
            "i8" => Some(Type::Int(IntType::I8)),

            "usize" => Some(Type::Int(IntType::USize)),
            "u128" => Some(Type::Int(IntType::U128)),
            "u64" => Some(Type::Int(IntType::U64)),
            "u32" => Some(Type::Int(IntType::U32)),
            "u16" => Some(Type::Int(IntType::U16)),
            "u8" => Some(Type::Int(IntType::U8)),

            "f64" => Some(Type::Float(FloatType::F64)),
            "f32" => Some(Type::Float(FloatType::F32)),

            "bool" => Some(Type::Bool),
            "char" => Some(Type::Char),

            _ => None,
        }
    }

    /// This will verify that the type is known, convert IntUnknown / FloatUnknown to their default types, or panic
    pub fn check_known(&mut self) -> Type {
        match self {
            Type::Int(_) | Type::Float(_) | Type::Bool | Type::Char | Type::Void | Type::Never => {
                ()
            }
            Type::Unknown => panic!("unknown type in function"),
            Type::IntUnknown => *self = Type::Int(IntType::I32),
            Type::FloatUnknown => *self = Type::Float(FloatType::F64),
            Type::Complex(ComplexType::Ref(inner, is_mut)) => {
                if inner.is_unknown() {
                    let new_inner = inner.clone().check_known();
                    *self = Type::from_complex(ComplexType::Ref(new_inner, *is_mut));
                }
            }
            Type::Complex(ComplexType::Ptr(inner, is_mut)) => {
                if inner.is_unknown() {
                    let new_inner = inner.clone().check_known();
                    *self = Type::from_complex(ComplexType::Ptr(new_inner, *is_mut));
                }
            }
        }
        *self
    }

    pub fn is_unknown(&self) -> bool {
        match self {
            Type::Int(_) | Type::Float(_) | Type::Bool | Type::Char | Type::Void | Type::Never => {
                false
            }
            Type::Unknown | Type::IntUnknown | Type::FloatUnknown => true,
            Type::Complex(cpx) => match cpx {
                ComplexType::Ref(t, _) | ComplexType::Ptr(t, _) => t.is_unknown(),
            },
        }
    }

    pub fn is_number(&self) -> bool {
        match self {
            Type::IntUnknown | Type::Int(_) | Type::FloatUnknown | Type::Float(_) => true,
            _ => false,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Type::IntUnknown | Type::Int(_) => true,
            _ => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Type::FloatUnknown | Type::Float(_) => true,
            _ => false,
        }
    }

    pub fn is_ptr(&self) -> bool {
        match self {
            Type::Complex(ComplexType::Ptr(..)) => true,
            _ => false,
        }
    }

    pub fn is_ref(&self) -> bool {
        match self {
            Type::Complex(ComplexType::Ref(..)) => true,
            _ => false,
        }
    }

    pub fn is_prim_eq(&self) -> bool {
        match self {
            Type::IntUnknown
            | Type::Int(_)
            | Type::FloatUnknown
            | Type::Float(_)
            | Type::Bool
            | Type::Char => true,
            _ => false,
        }
    }

    pub fn is_signed(&self) -> bool {
        if let Type::Int(ti) = self {
            match ti {
                IntType::ISize
                | IntType::I128
                | IntType::I64
                | IntType::I32
                | IntType::I16
                | IntType::I8 => true,
                IntType::USize
                | IntType::U128
                | IntType::U64
                | IntType::U32
                | IntType::U16
                | IntType::U8 => false,
            }
        } else if *self == Type::Char {
            false
        } else {
            panic!("can't check signed-ness of {:?}", self)
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Type::Int(IntType::I128) | Type::Int(IntType::U128) => 16,
            Type::Int(IntType::I64) | Type::Int(IntType::U64) => 8,
            Type::Int(IntType::I32) | Type::Int(IntType::U32) => 4,
            Type::Int(IntType::I16) | Type::Int(IntType::U16) => 2,
            Type::Int(IntType::I8) | Type::Int(IntType::U8) => 1,

            Type::Float(FloatType::F64) => 8,
            Type::Float(FloatType::F32) => 4,

            Type::Char => 4,

            Type::Int(IntType::ISize)
            | Type::Int(IntType::USize)
            | Type::Complex(ComplexType::Ptr(..))
            | Type::Complex(ComplexType::Ref(..)) => PTR_WIDTH,

            _ => panic!("cannot size {:?}", self),
        }
    }

    pub fn can_upgrade_to(self, other: Type) -> bool {
        if self == other {
            panic!("type equivilance should be checked before calling this");
        } else {
            match (self, other) {
                (Type::Unknown, _) => true,
                (_, Type::Unknown) => false,

                (Type::IntUnknown, Type::Int(_)) => true,
                (Type::Int(_), Type::IntUnknown) => false,

                (Type::FloatUnknown, Type::Float(_)) => true,
                (Type::Float(_), Type::FloatUnknown) => false,

                // we do not want to downgrade never to unknown
                // however, upgrading it to IntUnknown or FloatUnknown is fine
                (Type::Never, x) if x != Type::Unknown => true,
                (x, Type::Never) if x != Type::Unknown => false,

                (
                    Type::Complex(ComplexType::Ref(t1, m1)),
                    Type::Complex(ComplexType::Ref(t2, m2)),
                ) => {
                    assert_eq!(m1, m2);
                    t1.can_upgrade_to(*t2)
                }

                _ => panic!("type error, can not unify types {:?} and {:?}", self, other),
            }
        }
    }

    pub fn from_complex(t: ComplexType) -> Type {
        let mut registry = TYPE_REGISTRY.write().unwrap();
        if let Some(res) = registry.lookup.get(&t) {
            Type::Complex(res)
        } else {
            let res = Box::leak(Box::new(t));
            registry.lookup.insert(res);
            Type::Complex(res)
        }
    }

    // get target type of ref -- REF ONLY
    pub fn try_get_ref(self) -> Option<(Type, bool)> {
        if let Type::Complex(ComplexType::Ref(ty, is_mut)) = self {
            Some((*ty, *is_mut))
        } else {
            None
        }
    }
}
