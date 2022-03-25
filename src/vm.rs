use crate::{
    hir::{
        func::{Block, Expr, ExprInfo, FuncHIR},
        item::Function,
        types::{IntType, Type},
    },
    profiler::profile,
};

#[derive(Clone, Copy, Default)]
struct FrameAllocator(u32);

impl FrameAllocator {
    fn alloc(&mut self, ty: Type) -> Option<u32> {
        let size = ty.byte_size() as u32;
        if size == 0 {
            return None;
        }
        let align = ty.byte_align() as u32;
        while self.0 % align != 0 {
            self.0 += 1;
        }
        let offset = self.0;
        self.0 += size;
        Some(offset)
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
enum Instr {
    // out = const
    I32_Const(u32, i32),
    // out = i32
    Mov4(u32, u32),
    I32_Neg(u32, u32),
    I32_Not(u32, u32),
    // out = i32 <op> i32

    Eq4(u32, u32, u32),
    NotEq4(u32, u32, u32),

    I32_Add(u32, u32, u32),
    I32_Sub(u32, u32, u32),
    I32_Mul(u32, u32, u32),

    I32_Or(u32, u32, u32),
    I32_And(u32, u32, u32),
    I32_Xor(u32, u32, u32),
    
    I32_S_Div(u32, u32, u32),
    I32_S_Rem(u32, u32, u32),
    I32_S_Lt(u32, u32, u32),
    I32_S_LtEq(u32, u32, u32),
    // offset [cond slot]
    Jump(i32),
    JumpF(i32, u32),
    // no args
    Return,
    Bad,
    BuiltIn_print_i32(u32),
    BuiltIn_print_bool(u32)
}

pub fn exec(func: &Function) {
    let code = compile(func);
    let mut stack: Vec<u128> = vec![0, 1024];
    let stack = stack.as_mut_ptr() as *mut u8;
    unsafe {
        //profile("exec", || exec_rust(code, stack))
        profile("exec", || exec_rust(code, stack))
    }
}

unsafe fn write_stack<T>(base: *mut u8, offset: u32, x: T) {
    *(base.add(offset as usize) as *mut _) = x;
}

unsafe fn read_stack<T: Copy>(base: *mut u8, offset: u32) -> T {
    *(base.add(offset as usize) as *mut _)
}

unsafe fn exec_rust(code: Vec<Instr>, stack: *mut u8) {

    let mut pc = 0;

    loop {
        let instr = code[pc]; //*code.get_unchecked(pc);// [pc];
        match instr {
            Instr::Mov4(out, src) => {
                let x: i32 = read_stack(stack, src);
                write_stack(stack, out, x);
            }
            Instr::I32_Neg(out, src) => {
                let x: i32 = read_stack(stack, src);
                let res = x.wrapping_neg();
                write_stack(stack, out, res);
            }
            Instr::I32_Not(out, src) => {
                let x: i32 = read_stack(stack, src);
                let res = !x;
                write_stack(stack, out, res);
            }
            Instr::I32_Add(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: i32 = a.wrapping_add(b);
                write_stack(stack, out, res);
            }
            Instr::I32_Sub(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: i32 = a.wrapping_sub(b);
                write_stack(stack, out, res);
            }
            Instr::I32_Mul(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: i32 = a.wrapping_mul(b);
                write_stack(stack, out, res);
            }
            Instr::I32_Or(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: i32 = a | b;
                write_stack(stack, out, res);
            }
            Instr::I32_And(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: i32 = a & b;
                write_stack(stack, out, res);
            }
            Instr::I32_Xor(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: i32 = a ^ b;
                write_stack(stack, out, res);
            }
            Instr::I32_S_Div(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: i32 = a.wrapping_div(b);
                write_stack(stack, out, res);
            }
            Instr::I32_S_Rem(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: i32 = a.wrapping_rem(b);
                write_stack(stack, out, res);
            }
            Instr::I32_S_Lt(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: bool = a < b;
                write_stack(stack, out, res);
            }
            Instr::I32_S_LtEq(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: bool = a <= b;
                write_stack(stack, out, res);
            }
            Instr::Eq4(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: bool = a == b;
                write_stack(stack, out, res);
            }
            Instr::NotEq4(out, lhs, rhs) => {
                let a: i32 = read_stack(stack, lhs);
                let b: i32 = read_stack(stack, rhs);
                let res: bool = a != b;
                write_stack(stack, out, res);
            }
            Instr::I32_Const(out, n) => {
                write_stack(stack, out, n as i32);
            }
            Instr::JumpF(offset, cond) => {
                let x: bool = read_stack(stack, cond);
                if !x {
                    pc = (pc as isize + offset as isize) as usize;
                    continue;
                }
            }
            Instr::Jump(offset) => {
                pc = (pc as isize + offset as isize) as usize;
                continue;
            }
            Instr::Bad => panic!("encountered bad instruction"),
            Instr::Return => break,
            Instr::BuiltIn_print_i32(arg) => {
                let arg: i32 = read_stack(stack, arg);
                crate::builtin::print_i32(arg);
            },
            Instr::BuiltIn_print_bool(arg) => {
                let arg: bool = read_stack(stack, arg);
                crate::builtin::print_bool(arg);
            },
            //_ => panic!("todo execute {:?}", instr),
        }
        pc += 1;
    }
}

fn compile(func: &Function) -> Vec<Instr> {
    let sig = func.sig();
    let input_fn = func.hir();

    profile("lower HIR -> BC", || {
        let mut frame = FrameAllocator::default();
        let return_slot = frame.alloc(sig.output);

        match return_slot {
            Some(offset) => {
                assert_eq!(offset, 0);
            }
            None => (),
        }

        let mut var_map = vec!(None; input_fn.vars.len());
        
        for (i,input) in sig.inputs.iter().enumerate() {
            var_map[i] = frame.alloc(*input);
        }

        let mut compiler = BCompiler {
            var_map,
            code: Vec::new(),
            code_frame_depth: Vec::new(),
            input_fn,
            frame,
        };

        compiler.lower_expr(input_fn.root_expr as u32, return_slot);
        compiler.push_code(Instr::Return);
        if crate::VERBOSE {
            compiler.dump();
        }
        compiler.code
    })
}

struct BCompiler<'a> {
    var_map: Vec<Option<u32>>,
    code: Vec<Instr>,
    code_frame_depth: Vec<u32>,
    input_fn: &'a FuncHIR,
    frame: FrameAllocator,
}

fn instr_for_bin_op(op: syn::BinOp, arg_ty: Type) -> (fn(u32, u32, u32) -> Instr,bool) {
    match (op, arg_ty) {
        (syn::BinOp::Add(_), Type::Int(IntType::I32)) => (Instr::I32_Add,false),
        (syn::BinOp::Sub(_), Type::Int(IntType::I32)) => (Instr::I32_Sub,false),
        (syn::BinOp::Mul(_), Type::Int(IntType::I32)) => (Instr::I32_Mul,false),

        (syn::BinOp::BitOr(_), Type::Int(IntType::I32)) => (Instr::I32_Or,false),
        (syn::BinOp::BitAnd(_), Type::Int(IntType::I32)) => (Instr::I32_And,false),
        (syn::BinOp::BitXor(_), Type::Int(IntType::I32)) => (Instr::I32_Xor,false),

        (syn::BinOp::Eq(_), Type::Int(IntType::I32)) => (Instr::Eq4,false),
        (syn::BinOp::Ne(_), Type::Int(IntType::I32)) => (Instr::NotEq4,false),

        // sign-dependant
        (syn::BinOp::Div(_), Type::Int(IntType::I32)) => (Instr::I32_S_Div,false),
        (syn::BinOp::Rem(_), Type::Int(IntType::I32)) => (Instr::I32_S_Rem,false),
        (syn::BinOp::Lt(_), Type::Int(IntType::I32)) => (Instr::I32_S_Lt,false),
        (syn::BinOp::Le(_), Type::Int(IntType::I32)) => (Instr::I32_S_LtEq,false),
        (syn::BinOp::Gt(_), Type::Int(IntType::I32)) => (Instr::I32_S_Lt,true),
        (syn::BinOp::Ge(_), Type::Int(IntType::I32)) => (Instr::I32_S_LtEq,true),
        _ => panic!("todo bin-op {:?} {:?}", op, arg_ty),
    }
}

fn instr_for_un_op(op: syn::UnOp, ty: Type) -> fn(u32,u32) -> Instr {
    match (op,ty) {
        (syn::UnOp::Neg(_), Type::Int(IntType::I32)) => Instr::I32_Neg,
        (syn::UnOp::Not(_), Type::Int(IntType::I32)) => Instr::I32_Not,
        _ => panic!("todo un-op {:?} {:?}",op,ty)
    }
}

impl<'a> BCompiler<'a> {
    /*fn lower_expr_enforce_destination(&mut self, expr_id: u32, dest_slot: Option<u32>) {
        let dest_actual = self.lower_expr(expr_id,dest_slot);
        if dest_actual != dest_slot {
            let ty = self.input_fn.exprs[expr_id as usize].ty;
            self.insert_move(dest_slot.unwrap(),dest_actual.unwrap(),ty);
        }
    }*/
    fn push_code(&mut self, ins: Instr) {
        assert_eq!(self.code.len(), self.code_frame_depth.len());
        self.code.push(ins);
        self.code_frame_depth.push(self.frame.0);
    }

    fn lower_expr(&mut self, expr_id: u32, mandatory_dest_slot: Option<u32>) -> Option<u32> {
        let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[expr_id as usize];

        let saved_frame = self.frame;

        let res_slot = match expr {
            Expr::Var(id) => {
                let src = self.var_map[*id as usize];
                if mandatory_dest_slot.is_none() {
                    src
                } else {
                    self.insert_move(mandatory_dest_slot.unwrap(), src.unwrap(), *ty);
                    mandatory_dest_slot
                }
            }
            Expr::DeclVar(id) => {
                assert!(mandatory_dest_slot.is_none());
                let var_info = &self.input_fn.exprs[*id as usize];
                if let Expr::Var(var_index) = var_info.expr {
                    self.var_map[var_index as usize] = self.frame.alloc(var_info.ty);
                    return None;
                } else {
                    panic!("bad var decl");
                }
            }
            Expr::Assign(dest, src) => {
                assert!(mandatory_dest_slot.is_none());
                let dest_expr = &self.input_fn.exprs[*dest as usize].expr;
                if let Expr::Var(id) = dest_expr {
                    let sub_dest_slot = self.var_map[*id as usize];
                    self.lower_expr(*src, sub_dest_slot);
                } else {
                    panic!("non-trivial assign");
                }
                self.frame = saved_frame; // reset stack
                None
            }
            Expr::LitInt(n) => {
                let dest_slot = mandatory_dest_slot.or_else(|| self.frame.alloc(*ty));
                if let Some(dest_slot) = dest_slot {
                    match ty {
                        Type::Int(IntType::I32) | Type::Int(IntType::U32) => {
                            self.push_code(Instr::I32_Const(dest_slot, *n as i32));
                        }
                        _ => panic!("todo more literal ints"),
                    }
                }
                dest_slot
            }
            Expr::BinOpPrimitive(lhs, op, rhs) => {
                let arg_ty = self.input_fn.exprs[*lhs as usize].ty;

                let (ins_ctor,flip) = instr_for_bin_op(*op, arg_ty);

                let l_slot = self.lower_expr(*lhs, None).unwrap();
                let r_slot = self.lower_expr(*rhs, None).unwrap();

                self.frame = saved_frame; // args ready, reset stack
                let dest_slot = mandatory_dest_slot.or_else(|| self.frame.alloc(*ty));
                let ins = if !flip {
                    ins_ctor(dest_slot.unwrap(), l_slot, r_slot)
                } else {
                    ins_ctor(dest_slot.unwrap(), r_slot, l_slot)
                };
                self.push_code(ins);
                dest_slot
            }
            Expr::UnOpPrimitive(arg, op) => {
                let arg_slot = self.lower_expr(*arg, None).unwrap();
                let ins_ctor = instr_for_un_op(*op, *ty);
                self.frame = saved_frame; // arg ready, reset stack
                let dest_slot = mandatory_dest_slot.or_else(|| self.frame.alloc(*ty));
                let ins = ins_ctor(dest_slot.unwrap(), arg_slot);
                self.push_code(ins);
                dest_slot
            }
            Expr::CallBuiltin(name, args) => {
                if name == "print_i32" {
                    let arg_slot = self.lower_expr(args[0], None).unwrap();
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_i32(arg_slot));
                    None
                } else if name == "print_bool" {
                    let arg_slot = self.lower_expr(args[0], None).unwrap();
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_bool(arg_slot));
                    None
                } else {
                    panic!("unknown builtin");
                }
            }
            Expr::Block(block) => {
                self.lower_block(block, mandatory_dest_slot)
            }
            Expr::While(cond, body) => {
                assert!(mandatory_dest_slot.is_none());
                let pc_start = self.code.len() as i32;
                let cond_slot = self.frame.alloc(Type::Bool);
                self.lower_expr(*cond, cond_slot);
                let pc_cond_jump = self.code.len();
                self.frame = saved_frame; // cond_read, reset stack
                self.push_code(Instr::Bad);
                let block_res = self.lower_block(body, None);
                assert!(block_res.is_none());
                let pc_end = self.code.len() as i32;
                self.push_code(Instr::Jump(pc_start - pc_end));
                self.code[pc_cond_jump] =
                    Instr::JumpF(pc_end - pc_cond_jump as i32 + 1, cond_slot.unwrap());
                None
            }
            _ => panic!("vm compile {:?}", expr),
        };
        //self.frame = saved_frame;
        if mandatory_dest_slot.is_some() {
            assert_eq!(mandatory_dest_slot,res_slot);
        }

        res_slot
    }

    fn lower_block(&mut self, block: &Block, dest_slot: Option<u32>) -> Option<u32> {
        for expr_id in &block.stmts {
            self.lower_expr(*expr_id, None);
        }

        if let Some(expr_id) = &block.result {
            self.lower_expr(*expr_id, dest_slot)
        } else {
            None
        }
    }

    fn insert_move(&mut self, dest: u32, src: u32, ty: Type) {
        let size = ty.byte_size();
        let align = ty.byte_align();
        if size == 4 && align == 4 {
            self.push_code(Instr::Mov4(dest, src));
        } else {
            panic!("no move");
        }
    }

    fn dump(&self) {
        for ((i, ins), depth) in self.code.iter().enumerate().zip(self.code_frame_depth.iter()) {
            println!("{:4} {:4} {:?}", i, depth, ins);
        }
    }
}
