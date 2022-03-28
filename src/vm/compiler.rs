use crate::{
    hir::{
        func::{Block, Expr, ExprInfo, FuncHIR},
        item::Function,
        types::{IntType, Type},
    },
    profiler::profile,
};

use super::instr::Instr;

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

pub fn compile(func: &Function) -> Vec<Instr> {
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

        let mut var_map = vec![None; input_fn.vars.len()];

        for (i, input) in sig.inputs.iter().enumerate() {
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

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
enum BinOpInt {
    Add,
    Sub,
    Mul,
    Or,
    And,
    Xor,
    ShiftL,
    Eq,
    NotEq,

    S_Div,
    S_Rem,
    S_ShiftR,
    S_Lt,
    S_LtEq,

    U_Div,
    U_Rem,
    U_ShiftR,
    U_Lt,
    U_LtEq,
}

#[derive(PartialEq)]
enum BinOpFlag {
    None,
    SwapArgs,
    Assign,
}

fn bin_op_int(op: syn::BinOp, signed: bool) -> (BinOpInt, BinOpFlag) {
    match (op, signed) {
        (syn::BinOp::Add(_), _) => (BinOpInt::Add, BinOpFlag::None),
        (syn::BinOp::AddEq(_), _) => (BinOpInt::Add, BinOpFlag::Assign),
        (syn::BinOp::Sub(_), _) => (BinOpInt::Sub, BinOpFlag::None),
        (syn::BinOp::SubEq(_), _) => (BinOpInt::Sub, BinOpFlag::Assign),
        (syn::BinOp::Mul(_), _) => (BinOpInt::Mul, BinOpFlag::None),
        (syn::BinOp::MulEq(_), _) => (BinOpInt::Mul, BinOpFlag::Assign),
        (syn::BinOp::BitOr(_), _) => (BinOpInt::Or, BinOpFlag::None),
        (syn::BinOp::BitOrEq(_), _) => (BinOpInt::Or, BinOpFlag::Assign),
        (syn::BinOp::BitAnd(_), _) => (BinOpInt::And, BinOpFlag::None),
        (syn::BinOp::BitAndEq(_), _) => (BinOpInt::And, BinOpFlag::Assign),
        (syn::BinOp::BitXor(_), _) => (BinOpInt::Xor, BinOpFlag::None),
        (syn::BinOp::BitXorEq(_), _) => (BinOpInt::Xor, BinOpFlag::Assign),
        (syn::BinOp::Shl(_), _) => (BinOpInt::ShiftL, BinOpFlag::None),
        (syn::BinOp::ShlEq(_), _) => (BinOpInt::ShiftL, BinOpFlag::Assign),
        (syn::BinOp::Eq(_), _) => (BinOpInt::Eq, BinOpFlag::None),
        (syn::BinOp::Ne(_), _) => (BinOpInt::NotEq, BinOpFlag::None),

        (syn::BinOp::Div(_), true) => (BinOpInt::S_Div, BinOpFlag::None),
        (syn::BinOp::DivEq(_), true) => (BinOpInt::S_Div, BinOpFlag::Assign),
        (syn::BinOp::Rem(_), true) => (BinOpInt::S_Rem, BinOpFlag::None),
        (syn::BinOp::RemEq(_), true) => (BinOpInt::S_Rem, BinOpFlag::Assign),
        (syn::BinOp::Shr(_), true) => (BinOpInt::S_ShiftR, BinOpFlag::None),
        (syn::BinOp::ShrEq(_), true) => (BinOpInt::S_ShiftR, BinOpFlag::Assign),
        (syn::BinOp::Lt(_), true) => (BinOpInt::S_Lt, BinOpFlag::None),
        (syn::BinOp::Le(_), true) => (BinOpInt::S_LtEq, BinOpFlag::None),
        (syn::BinOp::Gt(_), true) => (BinOpInt::S_Lt, BinOpFlag::SwapArgs),
        (syn::BinOp::Ge(_), true) => (BinOpInt::S_LtEq, BinOpFlag::SwapArgs),

        (syn::BinOp::Div(_), false) => (BinOpInt::U_Div, BinOpFlag::None),
        (syn::BinOp::DivEq(_), false) => (BinOpInt::U_Div, BinOpFlag::Assign),
        (syn::BinOp::Rem(_), false) => (BinOpInt::U_Rem, BinOpFlag::None),
        (syn::BinOp::RemEq(_), false) => (BinOpInt::U_Rem, BinOpFlag::Assign),
        (syn::BinOp::Shr(_), false) => (BinOpInt::U_ShiftR, BinOpFlag::None),
        (syn::BinOp::ShrEq(_), false) => (BinOpInt::U_ShiftR, BinOpFlag::Assign),
        (syn::BinOp::Lt(_), false) => (BinOpInt::U_Lt, BinOpFlag::None),
        (syn::BinOp::Le(_), false) => (BinOpInt::U_LtEq, BinOpFlag::None),
        (syn::BinOp::Gt(_), false) => (BinOpInt::U_Lt, BinOpFlag::SwapArgs),
        (syn::BinOp::Ge(_), false) => (BinOpInt::U_LtEq, BinOpFlag::SwapArgs),

        (syn::BinOp::And(_), _) | (syn::BinOp::Or(_), _) => {
            panic!("can not apply logical op to integers")
        }
    }
}

fn instr_for_bin_op(op: syn::BinOp, arg_ty: Type) -> (fn(u32, u32, u32) -> Instr, BinOpFlag) {
    if arg_ty.is_int() {
        let (op, flag) = bin_op_int(op, arg_ty.is_signed());
        let width = arg_ty.byte_size();
        let instr = match (width, op) {
            (4, BinOpInt::Add) => Instr::I32_Add,
            (4, BinOpInt::Sub) => Instr::I32_Sub,
            (4, BinOpInt::Mul) => Instr::I32_Mul,
            (4, BinOpInt::Or) => Instr::I32_Or,
            (4, BinOpInt::And) => Instr::I32_And,
            (4, BinOpInt::Xor) => Instr::I32_Xor,
            (4, BinOpInt::ShiftL) => Instr::I32_ShiftL,
            (4, BinOpInt::Eq) => Instr::I32_Eq,
            (4, BinOpInt::NotEq) => Instr::I32_NotEq,
            (4, BinOpInt::S_Div) => Instr::I32_S_Div,
            (4, BinOpInt::S_Rem) => Instr::I32_S_Rem,
            (4, BinOpInt::S_ShiftR) => Instr::I32_S_ShiftR,
            (4, BinOpInt::S_Lt) => Instr::I32_S_Lt,
            (4, BinOpInt::S_LtEq) => Instr::I32_S_LtEq,
            (4, BinOpInt::U_Div) => Instr::I32_U_Div,
            (4, BinOpInt::U_Rem) => Instr::I32_U_Rem,
            (4, BinOpInt::U_ShiftR) => Instr::I32_U_ShiftR,
            (4, BinOpInt::U_Lt) => Instr::I32_U_Lt,
            (4, BinOpInt::U_LtEq) => Instr::I32_U_LtEq,

            (8, BinOpInt::Add) => Instr::I64_Add,
            (8, BinOpInt::Sub) => Instr::I64_Sub,
            (8, BinOpInt::Mul) => Instr::I64_Mul,
            (8, BinOpInt::Or) => Instr::I64_Or,
            (8, BinOpInt::And) => Instr::I64_And,
            (8, BinOpInt::Xor) => Instr::I64_Xor,
            (8, BinOpInt::ShiftL) => Instr::I64_ShiftL,
            (8, BinOpInt::Eq) => Instr::I64_Eq,
            (8, BinOpInt::NotEq) => Instr::I64_NotEq,
            (8, BinOpInt::S_Div) => Instr::I64_S_Div,
            (8, BinOpInt::S_Rem) => Instr::I64_S_Rem,
            (8, BinOpInt::S_ShiftR) => Instr::I64_S_ShiftR,
            (8, BinOpInt::S_Lt) => Instr::I64_S_Lt,
            (8, BinOpInt::S_LtEq) => Instr::I64_S_LtEq,
            (8, BinOpInt::U_Div) => Instr::I64_U_Div,
            (8, BinOpInt::U_Rem) => Instr::I64_U_Rem,
            (8, BinOpInt::U_ShiftR) => Instr::I64_U_ShiftR,
            (8, BinOpInt::U_Lt) => Instr::I64_U_Lt,
            (8, BinOpInt::U_LtEq) => Instr::I64_U_LtEq,
            _ => panic!("binop nyi {} {:?}", width, op),
        };

        (instr, flag)
    } else {
        panic!("todo more bin ops");
    }
}

fn instr_for_un_op(op: syn::UnOp, ty: Type) -> fn(u32, u32) -> Instr {
    match (op, ty) {
        (syn::UnOp::Neg(_), Type::Int(IntType::I32)) => Instr::I32_Neg,
        (syn::UnOp::Not(_), Type::Int(IntType::I32 | IntType::U32)) => Instr::I32_Not,

        (syn::UnOp::Neg(_), Type::Int(IntType::I64)) => Instr::I64_Neg,
        (syn::UnOp::Not(_), Type::Int(IntType::I64 | IntType::U64)) => Instr::I64_Not,
        _ => panic!("todo un-op {:?} {:?}", op, ty),
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

    fn get_assign_dest(&mut self, dest: u32, mandatory_dest_slot: Option<u32>) -> Option<u32> {
        assert!(mandatory_dest_slot.is_none());
        let dest_expr = &self.input_fn.exprs[dest as usize].expr;
        if let Expr::Var(id) = dest_expr {
            self.var_map[*id as usize]
        } else {
            panic!("non-trivial assign");
        }
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
                let assign_dest_slot = self.get_assign_dest(*dest, mandatory_dest_slot);
                self.lower_expr(*src, assign_dest_slot);
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
                        Type::Int(IntType::I64) | Type::Int(IntType::U64) => {
                            self.push_code(Instr::I64_Const(dest_slot, *n as i64));
                        }
                        _ => panic!("todo more literal ints"),
                    }
                }
                dest_slot
            }
            Expr::CastPrimitive(src) => {
                let src_ty = self.input_fn.exprs[*src as usize].ty;
                let res_ty = ty;

                if src_ty.is_int() && res_ty.is_int() {
                    let src_width = src_ty.byte_size();
                    let res_width = res_ty.byte_size();

                    if src_width < res_width {
                        let signed = src_ty.is_signed();
                        let src_slot = self.lower_expr(*src, None).unwrap();
                        let ins_ctor = match (src_width, res_width, signed) {
                            (4, 8, true) => Instr::I64_S_Widen_32,
                            (4, 8, false) => Instr::I64_U_Widen_32,
                            _ => panic!("todo widen {} {} {}", src_width, res_width, signed),
                        };
                        self.frame = saved_frame; // arg ready, reset stack
                        let dest_slot = mandatory_dest_slot.or_else(|| self.frame.alloc(*ty));
                        let ins = ins_ctor(dest_slot.unwrap(), src_slot);
                        self.push_code(ins);
                        dest_slot
                    } else {
                        // no-op
                        self.lower_expr(*src, mandatory_dest_slot)
                    }
                } else {
                    panic!("cast {:?} -> {:?}", src_ty, res_ty);
                }
            }
            Expr::BinOpPrimitive(lhs, op, rhs) => {
                let arg_ty = self.input_fn.exprs[*lhs as usize].ty;

                let (ins_ctor, flag) = instr_for_bin_op(*op, arg_ty);

                let l_slot = self.lower_expr(*lhs, None).unwrap();
                let r_slot = self.lower_expr(*rhs, None).unwrap();

                self.frame = saved_frame; // args ready, reset stack
                let dest_slot = if flag == BinOpFlag::Assign {
                    self.get_assign_dest(*lhs, mandatory_dest_slot)
                } else {
                    mandatory_dest_slot.or_else(|| self.frame.alloc(*ty))
                };

                let ins = if flag == BinOpFlag::SwapArgs {
                    ins_ctor(dest_slot.unwrap(), r_slot, l_slot)
                } else {
                    ins_ctor(dest_slot.unwrap(), l_slot, r_slot)
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
                if name == "print_i64" {
                    let arg_slot = self.lower_expr(args[0], None).unwrap();
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_i64(arg_slot));
                    None
                } else if name == "print_u64" {
                    let arg_slot = self.lower_expr(args[0], None).unwrap();
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_u64(arg_slot));
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
            Expr::Block(block) => self.lower_block(block, *ty, mandatory_dest_slot),
            Expr::While(cond, body) => {
                assert!(mandatory_dest_slot.is_none());
                let pc_start = self.code.len() as i32;
                let cond_slot = self.frame.alloc(Type::Bool);
                self.lower_expr(*cond, cond_slot);
                let pc_cond_jump = self.code.len();
                self.frame = saved_frame; // cond_read, reset stack
                self.push_code(Instr::Bad);
                let block_res = self.lower_block(body, Type::Void, None);
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
            assert_eq!(mandatory_dest_slot, res_slot);
        }

        res_slot
    }

    fn lower_block(
        &mut self,
        block: &Block,
        ty: Type,
        mandatory_dest_slot: Option<u32>,
    ) -> Option<u32> {
        let dest_slot = mandatory_dest_slot.or_else(|| self.frame.alloc(ty));

        let saved_frame = self.frame;
        for expr_id in &block.stmts {
            self.lower_expr(*expr_id, None);
        }

        if let Some(expr_id) = &block.result {
            self.lower_expr(*expr_id, dest_slot);
        }
        self.frame = saved_frame;
        dest_slot
    }

    fn insert_move(&mut self, dest: u32, src: u32, ty: Type) {
        let size = ty.byte_size();
        let align = ty.byte_align();
        if size == 8 && align == 8 {
            self.push_code(Instr::I64_Mov(dest, src));
        } else if size == 4 && align == 4 {
            self.push_code(Instr::I32_Mov(dest, src));
        } else {
            panic!("no move");
        }
    }

    fn dump(&self) {
        for ((i, ins), depth) in self
            .code
            .iter()
            .enumerate()
            .zip(self.code_frame_depth.iter())
        {
            println!("{:4} {:4} {:?}", i, depth, ins);
        }
    }
}
