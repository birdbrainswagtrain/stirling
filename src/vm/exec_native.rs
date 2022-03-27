use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi, x64::Assembler, DynamicLabel};

use crate::{profiler::profile, builtin::print_i32};

use super::instr::Instr;

extern "C" fn meme() {
    panic!("stop");
}

pub fn exec_native(code: Vec<Instr>, stack: *mut u8) {
    let (built_fn,size) = profile("lower BC -> ASM",|| {
        let mut ops = Assembler::new().unwrap();
        let fn_offset = ops.offset();
        
        // rbx = stack pointer
        dynasm!(ops
            ; mov rbx, rdi
            ; sub rsp, 8
        );
        
        let labels: Vec<_> = code.iter().map(|_| ops.new_dynamic_label()).collect();
        
        for (index,instr) in code.iter().enumerate() {
            let label = labels[index];
            dynasm!(ops; =>label);
            match instr {
                Instr::I32_Const(dst, n) => {
                    let dst = *dst as i32;
                    dynasm!(ops
                        ; mov DWORD [rbx+dst], *n
                    );
                }
                Instr::I32_Add(dst, lhs, rhs) => {
                    let lhs = *lhs as i32;
                    let rhs = *rhs as i32;
                    let dst = *dst as i32;

                    dynasm!(ops
                        ; mov eax, [rbx+lhs]
                        ; add eax, [rbx+rhs]
                        ; mov [rbx+dst], eax
                    );
                }
                Instr::I32_And(dst, lhs, rhs) => {
                    let lhs = *lhs as i32;
                    let rhs = *rhs as i32;
                    let dst = *dst as i32;

                    dynasm!(ops
                        ; mov eax, [rbx+lhs]
                        ; and eax, [rbx+rhs]
                        ; mov [rbx+dst], eax
                    );
                }
                Instr::I32_S_Lt(dst, lhs, rhs) => {
                    let lhs = *lhs as i32;
                    let rhs = *rhs as i32;
                    let dst = *dst as i32;

                    dynasm!(ops
                        ; mov eax, [rbx+lhs]
                        ; cmp eax, [rbx+rhs]
                        ; setl [rbx+dst]
                    );
                }
                Instr::BuiltIn_print_i32(src) => {
                    let src = *src as i32;
                    dynasm!(ops
                        ; mov rax, QWORD print_i32 as _
                        ; mov edi, [rbx+src]
                        ; call rax
                    );
                }
                Instr::JumpF(offset,src) => {
                    let target = labels[(index as isize + *offset as isize) as usize];
                    let src = *src as i32;
                    dynasm!(ops
                        ; mov al, [rbx+src]
                        ; test al,al
                        ; jz =>target
                    );
                }
                Instr::Jump(offset) => {
                    let target = labels[(index as isize + *offset as isize) as usize];
                    dynasm!(ops
                        ; jmp =>target
                    );
                }
                Instr::Return => {
                    dynasm!(ops
                        ; add rsp, 8
                        ; ret
                    );
                }
                _ => {
                    dynasm!(ops; ud2);
                    println!("todo instr {:?}",instr);
                }
            }
        }
        let end_offset = ops.offset();
        let size = end_offset.0 - fn_offset.0;
    
        let buf = ops.finalize().unwrap();
    
        let built_fn: extern "C" fn(*mut u8) = unsafe { std::mem::transmute(buf.ptr(fn_offset)) };
        std::mem::forget(buf);
        (built_fn,size)
    });

    /*let mut code = String::new();
    println!("size = {}",size);
    for i in 0..size {
        let d = unsafe {
            *(built_fn as *const u8).offset(i as isize)
        };
        code.push_str(&format!("{:02x}",d));
    }
    println!("{}",code);*/
    
    profile("exec hybrid",|| built_fn(stack));
}
