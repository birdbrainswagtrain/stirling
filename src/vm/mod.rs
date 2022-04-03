use crate::{
    hir::{
        item::Function,
    },
    USE_VM_NATIVE,
};

mod exec;
//mod exec_native;
mod compiler;
mod instr;

pub use instr::Instr;

pub fn exec_main(func: &Function) {
    let mut stack: Vec<u128> = vec![0; 1024];
    let stack = stack.as_mut_ptr() as *mut u8;

    exec(func,stack);
}

fn exec(func: &Function, stack: *mut u8) {

    let code = func.bytecode.get_or_init(|| compiler::compile(func));

    if USE_VM_NATIVE {
        //exec_native::exec_native(code, stack)
    } else {
        unsafe { exec::exec_rust(code, stack) }
    }
}
