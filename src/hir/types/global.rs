use std::{sync::{Arc, RwLock}, collections::HashSet};

use once_cell::sync::Lazy;

use crate::hir::item::{Scope, try_path_to_name};

use super::common::{TypeKind, IntWidth, FloatWidth, IntSign};

struct TypeRegistry {
    lookup: HashSet<Arc<[GlobalType]>>,
    empty: Arc<[GlobalType]>
}

static TYPE_REGISTRY: Lazy<RwLock<TypeRegistry>> = Lazy::new(|| {
    RwLock::new(TypeRegistry{
        lookup: HashSet::new(),
        empty: Vec::new().into_boxed_slice().into()
    })
});

#[derive(Debug,Clone,Hash,PartialEq,Eq)]
pub struct GlobalType {
    pub kind: TypeKind,
    pub args: Arc<[GlobalType]>
}

impl GlobalType {
    pub fn simple(kind: TypeKind) -> Self {
        let args = TYPE_REGISTRY.read().unwrap().empty.clone();
        GlobalType{kind, args}
    }

    pub fn with_args(kind: TypeKind, args: &[GlobalType]) -> Self {
        let mut registry = TYPE_REGISTRY.write().unwrap();
        let args = if let Some(res) = registry.lookup.get(args) {
            res.clone()
        } else {
            let args: Arc<[GlobalType]> = args.to_owned().into_boxed_slice().into();
            registry.lookup.insert(args.clone());
            args
        };
        GlobalType{kind, args}
    }

    pub fn from_syn(syn_ty: &syn::Type, scope: &Scope) -> GlobalType {
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
                let inner = GlobalType::from_syn(&tr.elem, scope);
                return GlobalType::with_args(TypeKind::Ref(is_mut),&[inner]);
            }
            syn::Type::Ptr(tp) => {
                let is_mut = tp.mutability.is_some();
                let inner = GlobalType::from_syn(&tp.elem, scope);
                //return Type::from_compound(CompoundType::Ptr(inner, is_mut));
                panic!()
            }
            syn::Type::Never(_) => return GlobalType::simple(TypeKind::Never),
            syn::Type::Infer(_) => return GlobalType::simple(TypeKind::Unknown),
            syn::Type::Tuple(syn::TypeTuple { elems, .. }) => {
                if elems.len() == 0 {
                    return GlobalType::simple(TypeKind::Tuple);
                } else {
                    panic!("todo real tuple types");
                }
            }
            _ => (),
        }
        panic!("failed to convert type {:?}", syn_ty)
    }

    pub fn from_str(name: &str) -> Option<GlobalType> {
        match name {
            "isize" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::IntSize,IntSign::Signed))))),
            "i128" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int128,IntSign::Signed))))),
            "i64" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int64,IntSign::Signed))))),
            "i32" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int32,IntSign::Signed))))),
            "i16" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int16,IntSign::Signed))))),
            "i8" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int8,IntSign::Signed))))),

            "usize" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::IntSize,IntSign::Unsigned))))),
            "u128" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int128,IntSign::Unsigned))))),
            "u64" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int64,IntSign::Unsigned))))),
            "u32" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int32,IntSign::Unsigned))))),
            "u16" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int16,IntSign::Unsigned))))),
            "u8" => Some(GlobalType::simple(TypeKind::Int(Some((IntWidth::Int8,IntSign::Unsigned))))),

            "f64" => Some(GlobalType::simple(TypeKind::Float(Some(FloatWidth::Float64)))),
            "f32" => Some(GlobalType::simple(TypeKind::Float(Some(FloatWidth::Float32)))),

            "bool" => Some(GlobalType::simple(TypeKind::Bool)),
            "char" => Some(GlobalType::simple(TypeKind::Char)),

            _ => None,
        }
    }

    pub fn byte_size(&self) -> usize {
        match self.kind {
            TypeKind::Int(Some((IntWidth::IntSize,_))) => 8,
            TypeKind::Int(Some((IntWidth::Int8,_))) => 1,
            TypeKind::Int(Some((IntWidth::Int16,_))) => 2,
            TypeKind::Int(Some((IntWidth::Int32,_))) => 4,
            TypeKind::Int(Some((IntWidth::Int64,_))) => 8,
            TypeKind::Int(Some((IntWidth::Int128,_))) => 16,

            TypeKind::Float(Some(FloatWidth::Float32)) => 4,
            TypeKind::Float(Some(FloatWidth::Float64)) => 8,

            TypeKind::Bool => 1,
            TypeKind::Char => 4,
            
            TypeKind::Never => 0,
            TypeKind::Tuple => {
                assert!(self.args.len()==0);
                0
            },
            TypeKind::Ref(_) => 8,
            _ => panic!("todo size {:?}",self.kind)
        }
    }

    pub fn byte_align(&self) -> usize {
        match self.kind {
            TypeKind::Int(Some((IntWidth::IntSize,_))) => 8,
            TypeKind::Int(Some((IntWidth::Int8,_))) => 1,
            TypeKind::Int(Some((IntWidth::Int16,_))) => 2,
            TypeKind::Int(Some((IntWidth::Int32,_))) => 4,
            TypeKind::Int(Some((IntWidth::Int64,_))) => 8,
            TypeKind::Int(Some((IntWidth::Int128,_))) => 16,

            TypeKind::Float(Some(FloatWidth::Float32)) => 4,
            TypeKind::Float(Some(FloatWidth::Float64)) => 8,

            TypeKind::Bool => 1,
            TypeKind::Char => 4,

            TypeKind::Never => 1,
            TypeKind::Tuple => {
                assert!(self.args.len()==0);
                1
            },
            TypeKind::Ref(_) => 8,
            _ => panic!("todo align {:?}",self.kind)
        }
    }

    pub fn is_signed(&self) -> bool {
        if let TypeKind::Int(Some((_,IntSign::Signed))) = self.kind {
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct Signature {
    pub inputs: Vec<GlobalType>,
    pub output: GlobalType,
}

impl Signature {
    pub fn new(inputs: Vec<GlobalType>, output: GlobalType) -> Self {
        Self { inputs, output }
    }

    pub fn from_syn(syn_sig: &syn::Signature, scope: &Scope) -> Signature {
        let mut inputs = Vec::new();
        for arg in &syn_sig.inputs {
            match arg {
                syn::FnArg::Receiver(_) => panic!("recv"),
                syn::FnArg::Typed(x) => {
                    inputs.push(GlobalType::from_syn(&x.ty, scope));
                }
            };
        }
        let output = match &syn_sig.output {
            syn::ReturnType::Default => GlobalType::simple(TypeKind::Tuple),
            syn::ReturnType::Type(_, syn_ty) => GlobalType::from_syn(&syn_ty, scope),
        };
        Signature { inputs, output }
    }
}
