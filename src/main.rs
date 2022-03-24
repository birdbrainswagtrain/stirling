mod builtin;
mod disassemble;
mod hir;
mod jit;
mod profiler;
mod vm;

use crate::hir::item::{Function, Item, ItemName, Scope};
use crate::jit::jit_compile;

use memoffset::offset_of;
use profiler::{profile, profile_log};
use vm::exec;

const PTR_WIDTH: usize = 8;

const VERBOSE: bool = true;
const LOG_JITS: bool = false;
const USE_VM: bool = true;

fn main() {
    check_abi();

    let file_name = std::env::args().nth(1).expect("no file name provided");

    let file_string = profile("load source", || {
        std::fs::read_to_string(&file_name).expect("failed to read source file")
    });

    let syn_tree: syn::File = profile("parse", || {
        syn::parse_str(&file_string).expect("failed to parse source code")
    });

    let module = Scope::from_syn_file(syn_tree);
    let module = module.borrow();

    if let Item::Fn(func) = module.get(&ItemName::Value("main".into())).unwrap() {
        if USE_VM {
            let res = exec(func, &[5, 10]);
            println!("res = {}", res);
        } else {
            jit_compile(func);

            let compiled_main = unsafe { std::mem::transmute::<_, fn()>(func.c_fn.get()) };
            profile("exec", || compiled_main());
        }
    } else {
        panic!("can't find main");
    };

    profile_log();
}

fn check_abi() {
    assert_eq!(std::mem::size_of::<usize>(), PTR_WIDTH);
    assert_eq!(offset_of!(Function, c_fn), 0);
}
