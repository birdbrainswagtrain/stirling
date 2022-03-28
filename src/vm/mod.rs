use crate::{
    hir::{
        func::{Block, Expr, ExprInfo, FuncHIR},
        item::Function,
        types::{IntType, Type},
    },
    profiler::profile,
    USE_VM_NATIVE,
};

mod exec;
//mod exec_native;
mod compiler;
mod instr;

use instr::Instr;

pub fn exec(func: &Function) {
    // sanity checks, todo move
    assert_eq!(std::mem::size_of::<instr::Instr>(),16);

    let code = compiler::compile(func);
    let mut stack: Vec<u128> = vec![0, 1024];
    let stack = stack.as_mut_ptr() as *mut u8;
    if USE_VM_NATIVE {
        //exec_native::exec_native(code, stack)
    } else {
        unsafe { exec::exec_rust(code, stack) }
    }
}
