
mod hir_items;
mod hir_expr;
mod hir_check;
mod types;
mod jit;
mod disassemble;


use std::time::Instant;

use crate::{hir_items::{Scope, ItemName, Item}};
use jit::JIT;

const VERBOSE: bool = false;

fn main() {
    let start = Instant::now();
    let file_string = std::fs::read_to_string("test/add.rs").expect("failed to read source file");

    let syn_tree: syn::File = syn::parse_str(&file_string).expect("failed to parse source code");

    let module = Scope::from_syn_file(syn_tree);
    let module = module.borrow();

    if let Item::Fn(func) = module.get(&ItemName::Value("add".into())).unwrap() {
        //let code = func.code();
        //code.print();
        let code = func.code();

        let mut jit: JIT = Default::default();

        let compiled_ptr = jit.compile(func.sig(),code).unwrap();
        
        println!("Build Time: {:?}",start.elapsed());

        let compiled_fn = unsafe { std::mem::transmute::<_, fn(i32,i32)->i32 >(compiled_ptr) };

        let start = Instant::now();
        let res = compiled_fn(10000,100_000_000);
        println!("Exec Time: {:?}",start.elapsed());
        println!("Result: {}",res);
    }
}
