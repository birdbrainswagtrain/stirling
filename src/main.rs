
mod hir_items;
mod hir_expr;
mod hir_check;
mod types;
mod jit;


use std::time::Instant;

use crate::{hir_items::{Scope, ItemName, Item}};
use jit::JIT;

const VERBOSE: bool = false;

fn main() {
    let start = Instant::now();
    let file_string = std::fs::read_to_string("test/add.rs").expect("failed to read source file");

    let syn_tree: syn::File = syn::parse_str(&file_string).expect("failed to parse source code");

    let scope = Scope::from_syn_file(syn_tree);
    let scope = scope.borrow();

    if let Item::Fn(func) = scope.get(&ItemName::Value("add".into())).unwrap() {
        //let code = func.code();
        //code.print();
        let code = func.code();

        let mut jit: JIT = Default::default();

        let (compiled_ptr,compiled_size) = jit.compile(func.sig(),code).unwrap();
        println!("Build Time: {:?}",start.elapsed());

        if VERBOSE {
            let compiled_slice = unsafe { std::slice::from_raw_parts(compiled_ptr,compiled_size) };
            disassemble(compiled_slice);
        }

        let compiled_fn = unsafe { std::mem::transmute::<_, fn(i32,i32)->i32 >(compiled_ptr) };

        let start = Instant::now();
        let res = compiled_fn(10000,100_000_000);
        println!("Exec Time: {:?}",start.elapsed());
        println!("Result: {}",res);
    }
}

fn disassemble(code: &[u8]) {
    use iced_x86::{Decoder,DecoderOptions,IntelFormatter,Formatter,Instruction};
    let mut decoder = Decoder::new(64, code, DecoderOptions::NONE);
    decoder.set_ip(0x1000);

    let mut formatter = IntelFormatter::new();

    let mut instruction = Instruction::default();
    let mut output = String::new();
    while decoder.can_decode() {
        output.clear();
        decoder.decode_out(&mut instruction);
        formatter.format(&instruction, &mut output);
        println!("  {:02x}  {}",instruction.ip(),output);
    }
}
