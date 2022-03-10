use crate::types::{Signature, Type, TypeInt};



#[derive(Debug,Clone,Copy)]
pub enum Builtin{
    PrintI32
}

impl Builtin {
    pub fn signature(&self) -> Signature {
        match self {
            Builtin::PrintI32 =>
                Signature::new(vec!(Type::Int(TypeInt::I32)), Type::Void)
        }
    }

    pub fn fn_ptr(&self) -> *const u8 {
        match self {
            Builtin::PrintI32 =>
                _builtin::print_i32 as _
        }
    }
}

mod _builtin {
    include!("../test/_skitter_builtin.rs");
}
