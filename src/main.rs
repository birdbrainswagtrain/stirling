mod builtin;
mod disassemble;
mod hir;
mod jit;

use crate::hir::item::{Function, Item, ItemName, Scope};
use crate::jit::jit_compile;

use memoffset::offset_of;

const PTR_WIDTH: usize = 8;
const VERBOSE: bool = true;

fn main() {
    check_abi();

    let file_string = std::fs::read_to_string("test/add.rs").expect("failed to read source file");

    let syn_tree: syn::File = syn::parse_str(&file_string).expect("failed to parse source code");

    let module = Scope::from_syn_file(syn_tree);
    let module = module.borrow();

    let compiled_main = if let Item::Fn(func) = module.get(&ItemName::Value("main".into())).unwrap()
    {
        jit_compile(func);

        unsafe { std::mem::transmute::<_, fn()>(func.c_fn.get()) }
    } else {
        panic!("can't find main");
    };

    compiled_main();
}

fn check_abi() {
    assert_eq!(std::mem::size_of::<usize>(), PTR_WIDTH);
    assert_eq!(offset_of!(Function, c_fn), 0);
}
