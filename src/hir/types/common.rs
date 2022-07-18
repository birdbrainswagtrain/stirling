#[derive(Debug,Clone,Copy,Hash,PartialEq,Eq)]
pub enum IntWidth {
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    IntSize
}

#[derive(Debug,Clone,Copy,Hash,PartialEq,Eq)]
pub enum FloatWidth {
    Float32,
    Float64
}

#[derive(Debug,Clone,Copy,Hash,PartialEq,Eq)]
pub enum IntSign {
    Signed,
    Unsigned
}

#[derive(Debug,Clone,Hash,PartialEq,Eq)]
pub enum TypeKind {
    Unknown,
    Never,
    Tuple, // used for '()' / void
    Int(Option<(IntWidth,IntSign)>),
    Float(Option<FloatWidth>),
    Bool,
    Char,
    Ref(bool) // arg = is mutable?
}

impl TypeKind {
    pub fn ptr() -> TypeKind {
        Self::Int(Some((IntWidth::IntSize,IntSign::Unsigned)))
    }

    pub fn is_known_strict(&self) -> bool {
        match self {
            TypeKind::Unknown | TypeKind::Int(None) | TypeKind::Float(None) => false,
            _ => true
        }
    }

    pub fn cannot_coerce(&self) -> bool {
        match self {
            // this should maybe be a blacklist, but use a whitelist for now
            TypeKind::Int(_) | TypeKind::Float(_) => true,
            _ => false
        }
    }

    /// Is the type a primitive for the purposes of operator inference?
    pub fn is_op_prim(&self) -> bool {
        match self {
            TypeKind::Bool | TypeKind::Int(_) | TypeKind::Float(_) => true,
            _ => false
        }
    }
}
