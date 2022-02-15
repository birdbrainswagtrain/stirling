
mod hir_items;
mod hir_expr;
mod types;

use hir_expr::FuncCode;

use crate::{hir_items::{Scope, ItemName, Item}, hir_expr::Block};

fn main() {
    let file_string = std::fs::read_to_string("test/add.rs").expect("failed to read source file");

    let syn_tree: syn::File = syn::parse_str(&file_string).expect("failed to parse source code");

    let scope = Scope::from_syn_file(syn_tree);
    let scope = scope.borrow();

    if let Item::Fn(func) = scope.get(&ItemName::Value("add".into())).unwrap() {
        //let code = func.code();
        //code.print();
        let x = func.sig();
        println!("{:?}",x);
    }
}
