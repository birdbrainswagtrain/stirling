
mod hir_items;
mod hir_expr;
mod types;
mod jit;


use crate::{hir_items::{Scope, ItemName, Item}};
use jit::JIT;

fn main() {
    let file_string = std::fs::read_to_string("test/add.rs").expect("failed to read source file");

    let syn_tree: syn::File = syn::parse_str(&file_string).expect("failed to parse source code");

    let scope = Scope::from_syn_file(syn_tree);
    let scope = scope.borrow();

    if let Item::Fn(func) = scope.get(&ItemName::Value("add".into())).unwrap() {
        //let code = func.code();
        //code.print();
        let code = func.code();
        code.print();

        let mut jit: JIT = Default::default();

        let compiled = jit.compile(func.sig(),code).unwrap();

        let compiled_fn = unsafe { std::mem::transmute::<_, fn(i32,i32)->i32 >(compiled) };

        println!("a {}",compiled_fn(111,222));
        //println!("=> {:?}",jit.compile(func.sig(),code));
    }
}
