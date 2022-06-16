mod builtin;
mod disassemble;
mod hir;
mod jit;
mod profiler;
mod vm;

use std::path::{Path, PathBuf};

use crate::hir::item::{Function, Item, ItemName, Scope};
use crate::jit::jit_compile;

use clap::Parser;
use memoffset::offset_of;
use once_cell::sync::OnceCell;
use profiler::{profile, profile_log};
use vm::exec_main;

const PTR_WIDTH: usize = 8;

static VERBOSE: OnceCell<bool> = OnceCell::new();
fn is_verbose() -> bool {
    *VERBOSE.get().unwrap()
}

const LOG_JITS: bool = false;
const USE_VM: bool = true;
const USE_VM_NATIVE: bool = false;

#[derive(clap::Parser, Debug)]
#[clap(author, version, about)]
struct CmdArgs{
    file_name: String,

    /// Used to run tests. Pass a directory as file_name.
    #[clap(long)]
    test: bool,

    /// Dump profiler information after running.
    #[clap(long)]
    profile: bool,

    /// Dumps debug information.
    #[clap(long,short)]
    verbose: bool
}

fn main() {
    check_abi();

    let args = CmdArgs::parse();

    VERBOSE.set(args.verbose).unwrap();
    
    if args.test {
        test(&args.file_name);
    }

    //let file_name = std::env::args().nth(1).expect("no file name provided");

    let file_string = profile("load source", || {
        std::fs::read_to_string(&args.file_name).expect("failed to read source file")
    });

    let syn_tree: syn::File = profile("parse", || {
        syn::parse_str(&file_string).expect("failed to parse source code")
    });

    let module = Scope::from_syn_file(syn_tree);
    let module = module.borrow();

    if let Item::Fn(func) = module.get(&ItemName::Value("main".into())).unwrap() {
        if USE_VM {
            exec_main(func);
        } else {
            jit_compile(func);

            let compiled_main = unsafe { std::mem::transmute::<_, fn()>(func.c_fn.get()) };
            profile("exec compiled", || compiled_main());
        }
    } else {
        panic!("can't find main");
    };

    if args.profile {
        profile_log();
    }
}

fn check_abi() {
    assert_eq!(std::mem::size_of::<usize>(), PTR_WIDTH);
    assert_eq!(offset_of!(Function, c_fn), 0);
    assert_eq!(std::mem::size_of::<crate::vm::Instr>(),16);
}

fn gather_tests(dir_name: &str, files: &mut Vec<PathBuf>) {
    let dir_path = Path::new(dir_name);
    let read_dir = std::fs::read_dir(dir_name).expect("failed to read test directory");
    for entry in read_dir {
        if let Ok(entry) = entry {
            if let Some(file_name) = entry.file_name().to_str() {
                if file_name.starts_with('_') {
                    continue;
                }
                if let Ok(file_ty) = entry.file_type() {
                    if file_ty.is_dir() {
                        let sub_dir = dir_path.join(file_name);
                        if let Some(sub_dir) = sub_dir.to_str() {
                            gather_tests(sub_dir,files);
                        }
                    } else if file_ty.is_file() {
                        if file_name.ends_with(".rs") {
                            let file = dir_path.join(file_name);
                            files.push(file);
                        }
                    }
                }
            }
        }
    }
}

fn test(dir_name: &str) -> ! {
    use colored::Colorize;

    let mut test_files = Vec::new();
    gather_tests(dir_name,&mut test_files);
    test_files.sort();
    
    let bin_name = Path::new("/tmp/stirling_test_1");
    for file in test_files {
        let res = run_test(&file,bin_name);
        let res_str = if let Err(msg) = res {
            format!("FAIL: {}",msg).red()
        } else {
            "OKAY".green()
        };
        println!("{:35} {}",file.to_str().unwrap(),res_str);
    }

    profile_log();
    
    std::process::exit(0)
}

fn run_test(file_name: &Path, bin_name: &Path) -> Result<(),String> {
    use std::process::Command;
    // Rust compile
    profile("rustc compile",||{
        let fail = || Err(String::from("rustc compile failed"));

        let cmd_res = Command::new("rustc")
            .arg(file_name)
            .arg("-o").arg(bin_name)
            .arg("-C").arg("overflow-checks=off")
            .output();

        if let Ok(cmd_res) = cmd_res {
            if !cmd_res.status.success() {
                fail()
            } else {
                Ok(())
            }
        } else {
            fail()
        }
    })?;

    let rustc_out = profile("rustc exec",||{
        let fail = || Err(String::from("rustc exec failed"));

        let cmd_res = Command::new(bin_name).output();
        if let Ok(cmd_res) = cmd_res {
            if !cmd_res.status.success() {
                fail()
            } else {
                Ok(cmd_res.stdout)
            }
        } else {
            fail()
        }
    })?;

    let stirling_out = profile("stirling",||{
        let fail = || Err(String::from("stirling failed"));

        let program = std::env::current_exe().expect("failed to get stirling path");

        let cmd_res = Command::new(program)
            .arg(file_name)
            .output();

        if let Ok(cmd_res) = cmd_res {
            if !cmd_res.status.success() {
                fail()
            } else {
                Ok(cmd_res.stdout)
            }
        } else {
            fail()
        }
    })?;

    if rustc_out != stirling_out {
        Err("rustc and stirling output mismatch".into())
    } else {
        Ok(())
    }
}
