use once_cell::sync::{Lazy, OnceCell};

use std::{collections::HashSet, sync::RwLock};

use crate::PTR_WIDTH;

use super::item::{try_path_to_name, Scope};

struct TypeRegistry {
    lookup: HashSet<&'static CompoundType>,
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
pub enum CompoundType {
    Ref(Type, bool),
    Ptr(Type, bool),
    Tuple(Vec<Type>,InvisibleMetadata<OnceCell<StructLayout>>),
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
    Compound(&'static CompoundType),
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
                return Type::from_compound(CompoundType::Ref(inner, is_mut));
            }
            syn::Type::Ptr(tp) => {
                let is_mut = tp.mutability.is_some();
                let inner = Type::from_syn(&tp.elem, scope);
                return Type::from_compound(CompoundType::Ptr(inner, is_mut));
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
            Type::Compound(CompoundType::Ref(inner, is_mut)) => {
                if inner.is_unknown() {
                    let new_inner = inner.clone().check_known();
                    *self = Type::from_compound(CompoundType::Ref(new_inner, *is_mut));
                }
            }
            Type::Compound(CompoundType::Ptr(inner, is_mut)) => {
                if inner.is_unknown() {
                    let new_inner = inner.clone().check_known();
                    *self = Type::from_compound(CompoundType::Ptr(new_inner, *is_mut));
                }
            }
            Type::Compound(CompoundType::Tuple(fields,_)) => {
                if self.is_unknown() {
                    let mut fields = fields.clone();
                    for field in fields.iter_mut() {
                        field.check_known();
                    }
                    *self = Type::from_compound(CompoundType::Tuple(fields,Default::default()));
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
            Type::Compound(cpx) => match cpx {
                CompoundType::Ref(t, _) | CompoundType::Ptr(t, _) => t.is_unknown(),
                CompoundType::Tuple(fields,_) => fields.iter().any(|x| x.is_unknown()),
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
            Type::Compound(CompoundType::Ptr(..)) => true,
            _ => false,
        }
    }

    pub fn is_ref(&self) -> bool {
        match self {
            Type::Compound(CompoundType::Ref(..)) => true,
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
        } else if *self == Type::Char || *self == Type::Bool {
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
            Type::Bool => 1,
            Type::Void => 0,
            Type::Never => 0,

            Type::Int(IntType::ISize)
            | Type::Int(IntType::USize)
            | Type::Compound(CompoundType::Ptr(..))
            | Type::Compound(CompoundType::Ref(..)) => PTR_WIDTH,

            Type::Compound(CompoundType::Tuple(members,layout)) => {
                layout.as_ref().get_or_init(|| StructLayout::new(members)).size
            }

            _ => panic!("cannot size {:?}", self),
        }
    }

    pub fn byte_align(&self) -> usize {
        match self {
            Type::Int(IntType::I128) | Type::Int(IntType::U128) => 16,
            Type::Int(IntType::I64) | Type::Int(IntType::U64) => 8,
            Type::Int(IntType::I32) | Type::Int(IntType::U32) => 4,
            Type::Int(IntType::I16) | Type::Int(IntType::U16) => 2,
            Type::Int(IntType::I8) | Type::Int(IntType::U8) => 1,

            Type::Float(FloatType::F64) => 8,
            Type::Float(FloatType::F32) => 4,

            Type::Char => 4,
            Type::Bool => 1,
            Type::Void => 0,
            Type::Never => 0,

            Type::Int(IntType::ISize)
            | Type::Int(IntType::USize)
            | Type::Compound(CompoundType::Ptr(..))
            | Type::Compound(CompoundType::Ref(..)) => PTR_WIDTH,

            Type::Compound(CompoundType::Tuple(members,layout)) => {
                layout.as_ref().get_or_init(|| StructLayout::new(members)).align
            }

            _ => panic!("cannot align {:?}", self),
        }
    }

    pub fn layout(&self) -> &StructLayout {
        match self {
            Type::Compound(CompoundType::Tuple(members,layout)) => {
                layout.as_ref().get_or_init(|| StructLayout::new(members))
            }
            _ => panic!("can't get layout for {:?}",self)
        }
    }

    pub fn unify(self, other: Type) -> Type {
        if self == other {
            panic!("check type equality before calling this")
        } else {
            match (self, other) {
                (_, Type::Unknown) => self,
                (Type::Unknown, _) => other,

                (Type::Int(_), Type::IntUnknown) => self,
                (Type::IntUnknown, Type::Int(_)) => other,

                (Type::Float(_), Type::FloatUnknown) => self,
                (Type::FloatUnknown, Type::Float(_)) => other,

                // upgrade all types from never
                (_, Type::Never) => self,
                (Type::Never, _) => other,
                //(Type::Never, x) if x != Type::Unknown => other,

                (
                    Type::Compound(CompoundType::Ref(t1, m1)),
                    Type::Compound(CompoundType::Ref(t2, m2)),
                ) => {
                    let m = *m1 | *m2;
                    if t1 != t2 {
                        let unified = t1.unify(*t2);
                        Type::from_compound(CompoundType::Ref(unified, m))
                    } else {
                        Type::from_compound(CompoundType::Ref(*t1, m))
                    }
                }

                (
                    Type::Compound(CompoundType::Tuple(t1, m1)),
                    Type::Compound(CompoundType::Tuple(t2, m2)),
                ) => {
                    assert_eq!(t1.len(),t2.len());

                    let merged: Vec<Type> = t1.iter().zip(t2.iter()).map(|(t1,t2)| {
                        if t1 == t2 {
                            *t1
                        } else {
                            t1.unify(*t2)
                        }
                    }).collect();

                    Type::from_compound(CompoundType::Tuple(merged, Default::default()))
                }

                _ => panic!("type error, can not unify types {:?} and {:?}", self, other),
            }
        }
    }

    pub fn from_compound(t: CompoundType) -> Type {
        let mut registry = TYPE_REGISTRY.write().unwrap();
        if let Some(res) = registry.lookup.get(&t) {
            Type::Compound(res)
        } else {
            let res = Box::leak(Box::new(t));
            registry.lookup.insert(res);
            Type::Compound(res)
        }
    }

    pub fn get_tuple_member(self, index: u32) -> Option<Type> {
        match self {
            Type::Compound(CompoundType::Ref(child,_)) => child.get_tuple_member(index),
            Type::Compound(CompoundType::Tuple(members,_)) => Some(members[index as usize]),
            _ => None
        }
    }

    pub fn set_tuple_member(self, index: u32, ty: Type) -> Option<Type> {
        match self {
            Type::Compound(CompoundType::Ref(child,is_mut)) => {
                let child = child.set_tuple_member(index,ty)?;
                Some(Type::from_compound(CompoundType::Ref(child,*is_mut)))
            }
            Type::Compound(CompoundType::Tuple(members,_)) => {
                let mut new_members = members.clone();
                new_members[index as usize] = ty;
                Some(Type::from_compound(CompoundType::Tuple(new_members,Default::default())))
            },
            _ => None
        }
    }

    pub fn get_referenced_r(self) -> Type {
        match self {
            Type::Compound(CompoundType::Ref(child,is_mut)) => {
                child.get_referenced_r()
            }
            _ => self
        }
    }
}


/// HACK: Used to attach metadata (such as layout) to a type without making it affect hashing or equality.
#[derive(Debug,Default)]
pub struct InvisibleMetadata<T>(T);

impl<T> AsRef<T> for InvisibleMetadata<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T> Eq for InvisibleMetadata<T> {}

impl<T> PartialEq for InvisibleMetadata<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T> std::hash::Hash for InvisibleMetadata<T> {
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {
        // no-op
    }
}

#[derive(Debug)]
pub struct StructLayout{
    pub size: usize,
    pub align: usize,
    pub member_offsets: Vec<u32>,
}

impl StructLayout{
    fn new(members: &Vec<Type>) -> Self {
        let mut layout = Self{
            size: 0,
            align: 1,
            member_offsets: vec!()
        };

        for member in members {
            let m_size = member.byte_size();
            let m_align = member.byte_align();

            layout.align = layout.align.max(m_align);

            while layout.size % m_align != 0 {
                layout.size += 1;
            }

            layout.member_offsets.push(layout.size as u32);

            layout.size += m_size;
        }

        while layout.size % layout.align != 0 {
            layout.size += 1;
        }

        layout
    }
}
