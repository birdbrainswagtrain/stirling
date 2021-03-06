
use crate::{
    hir::{
        func::{Block, Expr, ExprInfo, FuncHIR},
        item::Function
    },
    is_verbose,
    profiler::profile,
};

use crate::hir::types::global::GlobalType;
use crate::hir::types::common::{TypeKind, IntWidth, IntSign, FloatWidth};

use super::instr::Instr;

#[derive(PartialEq)]
enum LoopJumpKind {
    Break,
    Continue,
}

struct LoopJump {
    kind: LoopJumpKind,
    loop_id: u32,
    instr_index: usize,
}

struct LoopResult {
    loop_id: u32,
    slot: u32,
}

#[derive(Clone, Copy, Default, PartialEq)]
struct FrameAllocator(u32);

impl FrameAllocator {
    fn alloc(&mut self, ty: &GlobalType) -> u32 {
        let size = ty.byte_size() as u32;
        let align = ty.byte_align() as u32;
        while self.0 % align != 0 {
            self.0 += 1;
        }
        let offset = self.0;
        self.0 += size;
        offset
    }

    fn align_for_call(&mut self) -> u32 {
        let align = 16;
        while self.0 % align != 0 {
            self.0 += 1;
        }
        self.0
    }
}

pub fn compile(func: &Function) -> Vec<Instr> {
    let sig = func.sig();
    let input_fn = func.hir();

    let (res,_) = profile("lower HIR -> BC", || {
        let mut frame = FrameAllocator::default();
        let return_slot = frame.alloc(&GlobalType::simple(TypeKind::ptr()));

        assert_eq!(return_slot, 0);

        let mut var_map = vec![None; input_fn.vars.len()];

        for (i, input) in sig.inputs.iter().enumerate() {
            var_map[i] = Some(frame.alloc(input));
        }

        let mut compiler = BCompiler {
            var_map,
            code: Vec::new(),
            code_frame_depth: Vec::new(),
            input_fn,
            frame,
            temp_slots: Vec::new(),
            loop_jumps: Vec::new(),
            loop_results: Vec::new(),
        };

        let res = compiler.lower_expr(input_fn.root_expr as u32, None);

        // add return
        compiler.insert_return(res);

        if is_verbose() {
            compiler.dump();
        }
        assert_eq!(compiler.loop_jumps.len(), 0);
        compiler.code
    });

    res
}

struct BCompiler<'a> {
    var_map: Vec<Option<u32>>,
    code: Vec<Instr>,
    code_frame_depth: Vec<u32>,
    input_fn: &'a FuncHIR,
    frame: FrameAllocator,
    temp_slots: Vec<(u32, u32)>,
    loop_jumps: Vec<LoopJump>,
    loop_results: Vec<LoopResult>,
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

#[derive(Debug, Clone, Copy)]
enum BinOpFloat {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

#[derive(PartialEq)]
enum BinOpFlag {
    None,
    SwapArgs,
    Assign,
}

fn bin_op_float(op: syn::BinOp) -> (BinOpFloat, BinOpFlag) {
    match op {
        syn::BinOp::Add(_) => (BinOpFloat::Add, BinOpFlag::None),
        syn::BinOp::AddEq(_) => (BinOpFloat::Add, BinOpFlag::Assign),
        syn::BinOp::Sub(_) => (BinOpFloat::Sub, BinOpFlag::None),
        syn::BinOp::SubEq(_) => (BinOpFloat::Sub, BinOpFlag::Assign),
        syn::BinOp::Mul(_) => (BinOpFloat::Mul, BinOpFlag::None),
        syn::BinOp::MulEq(_) => (BinOpFloat::Mul, BinOpFlag::Assign),
        syn::BinOp::Div(_) => (BinOpFloat::Div, BinOpFlag::None),
        syn::BinOp::DivEq(_) => (BinOpFloat::Div, BinOpFlag::Assign),
        syn::BinOp::Rem(_) => (BinOpFloat::Rem, BinOpFlag::None),
        syn::BinOp::RemEq(_) => (BinOpFloat::Rem, BinOpFlag::Assign),

        syn::BinOp::Eq(_) => (BinOpFloat::Eq, BinOpFlag::None),
        syn::BinOp::Ne(_) => (BinOpFloat::NotEq, BinOpFlag::None),
        syn::BinOp::Lt(_) => (BinOpFloat::Lt, BinOpFlag::None),
        syn::BinOp::Le(_) => (BinOpFloat::LtEq, BinOpFlag::None),
        syn::BinOp::Gt(_) => (BinOpFloat::Gt, BinOpFlag::None),
        syn::BinOp::Ge(_) => (BinOpFloat::GtEq, BinOpFlag::None),
        _ => panic!("can not apply {:?} to floats", op),
    }
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

fn instr_for_bin_op(op: syn::BinOp, arg_ty: &GlobalType) -> (fn(u32, u32, u32) -> Instr, BinOpFlag) {
    match &arg_ty.kind {
        TypeKind::Int(_) | TypeKind::Bool | TypeKind::Char => {
            let (op, flag) = bin_op_int(op, arg_ty.is_signed());
            let width = arg_ty.byte_size();

            let instr = match (width, op) {
                (1, BinOpInt::Add) => Instr::I8_Add,
                (1, BinOpInt::Sub) => Instr::I8_Sub,
                (1, BinOpInt::Mul) => Instr::I8_Mul,
                (1, BinOpInt::Or) => Instr::I8_Or,
                (1, BinOpInt::And) => Instr::I8_And,
                (1, BinOpInt::Xor) => Instr::I8_Xor,
                (1, BinOpInt::ShiftL) => Instr::I8_ShiftL,
                (1, BinOpInt::Eq) => Instr::I8_Eq,
                (1, BinOpInt::NotEq) => Instr::I8_NotEq,
                (1, BinOpInt::S_Div) => Instr::I8_S_Div,
                (1, BinOpInt::S_Rem) => Instr::I8_S_Rem,
                (1, BinOpInt::S_ShiftR) => Instr::I8_S_ShiftR,
                (1, BinOpInt::S_Lt) => Instr::I8_S_Lt,
                (1, BinOpInt::S_LtEq) => Instr::I8_S_LtEq,
                (1, BinOpInt::U_Div) => Instr::I8_U_Div,
                (1, BinOpInt::U_Rem) => Instr::I8_U_Rem,
                (1, BinOpInt::U_ShiftR) => Instr::I8_U_ShiftR,
                (1, BinOpInt::U_Lt) => Instr::I8_U_Lt,
                (1, BinOpInt::U_LtEq) => Instr::I8_U_LtEq,
    
                (2, BinOpInt::Add) => Instr::I16_Add,
                (2, BinOpInt::Sub) => Instr::I16_Sub,
                (2, BinOpInt::Mul) => Instr::I16_Mul,
                (2, BinOpInt::Or) => Instr::I16_Or,
                (2, BinOpInt::And) => Instr::I16_And,
                (2, BinOpInt::Xor) => Instr::I16_Xor,
                (2, BinOpInt::ShiftL) => Instr::I16_ShiftL,
                (2, BinOpInt::Eq) => Instr::I16_Eq,
                (2, BinOpInt::NotEq) => Instr::I16_NotEq,
                (2, BinOpInt::S_Div) => Instr::I16_S_Div,
                (2, BinOpInt::S_Rem) => Instr::I16_S_Rem,
                (2, BinOpInt::S_ShiftR) => Instr::I16_S_ShiftR,
                (2, BinOpInt::S_Lt) => Instr::I16_S_Lt,
                (2, BinOpInt::S_LtEq) => Instr::I16_S_LtEq,
                (2, BinOpInt::U_Div) => Instr::I16_U_Div,
                (2, BinOpInt::U_Rem) => Instr::I16_U_Rem,
                (2, BinOpInt::U_ShiftR) => Instr::I16_U_ShiftR,
                (2, BinOpInt::U_Lt) => Instr::I16_U_Lt,
                (2, BinOpInt::U_LtEq) => Instr::I16_U_LtEq,
    
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
    
                (16, BinOpInt::Add) => Instr::I128_Add,
                (16, BinOpInt::Sub) => Instr::I128_Sub,
                (16, BinOpInt::Mul) => Instr::I128_Mul,
                (16, BinOpInt::Or) => Instr::I128_Or,
                (16, BinOpInt::And) => Instr::I128_And,
                (16, BinOpInt::Xor) => Instr::I128_Xor,
                (16, BinOpInt::ShiftL) => Instr::I128_ShiftL,
                (16, BinOpInt::Eq) => Instr::I128_Eq,
                (16, BinOpInt::NotEq) => Instr::I128_NotEq,
                (16, BinOpInt::S_Div) => Instr::I128_S_Div,
                (16, BinOpInt::S_Rem) => Instr::I128_S_Rem,
                (16, BinOpInt::S_ShiftR) => Instr::I128_S_ShiftR,
                (16, BinOpInt::S_Lt) => Instr::I128_S_Lt,
                (16, BinOpInt::S_LtEq) => Instr::I128_S_LtEq,
                (16, BinOpInt::U_Div) => Instr::I128_U_Div,
                (16, BinOpInt::U_Rem) => Instr::I128_U_Rem,
                (16, BinOpInt::U_ShiftR) => Instr::I128_U_ShiftR,
                (16, BinOpInt::U_Lt) => Instr::I128_U_Lt,
                (16, BinOpInt::U_LtEq) => Instr::I128_U_LtEq,
    
                _ => panic!("binop nyi {} {:?}", width, op),
            };
    
            (instr, flag)
        },
        TypeKind::Float(_) => {
            let (op, flag) = bin_op_float(op);
            let instr = match (&arg_ty.kind, op) {
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::Add) => Instr::F32_Add,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::Sub) => Instr::F32_Sub,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::Mul) => Instr::F32_Mul,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::Div) => Instr::F32_Div,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::Rem) => Instr::F32_Rem,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::Eq) => Instr::F32_Eq,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::NotEq) => Instr::F32_NotEq,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::Lt) => Instr::F32_Lt,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::LtEq) => Instr::F32_LtEq,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::Gt) => Instr::F32_Gt,
                (TypeKind::Float(Some(FloatWidth::Float32)), BinOpFloat::GtEq) => Instr::F32_GtEq,

                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::Add) => Instr::F64_Add,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::Sub) => Instr::F64_Sub,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::Mul) => Instr::F64_Mul,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::Div) => Instr::F64_Div,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::Rem) => Instr::F64_Rem,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::Eq) => Instr::F64_Eq,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::NotEq) => Instr::F64_NotEq,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::Lt) => Instr::F64_Lt,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::LtEq) => Instr::F64_LtEq,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::Gt) => Instr::F64_Gt,
                (TypeKind::Float(Some(FloatWidth::Float64)), BinOpFloat::GtEq) => Instr::F64_GtEq,

                _ => panic!("binop nyi {:?} {:?}", arg_ty, op)
            };

            (instr,flag)
        }
        _ => panic!("todo binop {:?}",arg_ty)
    }
}

fn instr_for_un_op(op: syn::UnOp, ty: &GlobalType) -> fn(u32, u32) -> Instr {
    match (op,&ty.kind) {
        // integer operations
        (syn::UnOp::Neg(_),TypeKind::Int(Some((IntWidth::Int128,_)))) => Instr::I128_Neg,
        (syn::UnOp::Not(_),TypeKind::Int(Some((IntWidth::Int128,_)))) => Instr::I128_Not,
        (syn::UnOp::Neg(_),TypeKind::Int(Some((IntWidth::Int64,_)))) => Instr::I64_Neg,
        (syn::UnOp::Not(_),TypeKind::Int(Some((IntWidth::Int64,_)))) => Instr::I64_Not,
        (syn::UnOp::Neg(_),TypeKind::Int(Some((IntWidth::Int32,_)))) => Instr::I32_Neg,
        (syn::UnOp::Not(_),TypeKind::Int(Some((IntWidth::Int32,_)))) => Instr::I32_Not,
        (syn::UnOp::Neg(_),TypeKind::Int(Some((IntWidth::Int16,_)))) => Instr::I16_Neg,
        (syn::UnOp::Not(_),TypeKind::Int(Some((IntWidth::Int16,_)))) => Instr::I16_Not,
        (syn::UnOp::Neg(_),TypeKind::Int(Some((IntWidth::Int8,_)))) => Instr::I8_Neg,
        (syn::UnOp::Not(_),TypeKind::Int(Some((IntWidth::Int8,_)))) => Instr::I8_Not,
        // isize
        (syn::UnOp::Neg(_),TypeKind::Int(Some((IntWidth::IntSize,_)))) => Instr::I64_Neg,
        (syn::UnOp::Not(_),TypeKind::Int(Some((IntWidth::IntSize,_)))) => Instr::I64_Not,

        (syn::UnOp::Neg(_),TypeKind::Float(Some(FloatWidth::Float32))) => Instr::F32_Neg,
        (syn::UnOp::Neg(_),TypeKind::Float(Some(FloatWidth::Float64))) => Instr::F64_Neg,

        (syn::UnOp::Not(_),TypeKind::Bool) => Instr::Bool_Not,

        
        _ => panic!("todo un-op {:?} {:?}",op,ty)
    }
    /*
    } else if ty.is_float() {
        match (op, ty) {
            (syn::UnOp::Neg(_), Type::Float(FloatType::F64)) => Instr::F64_Neg,
            (syn::UnOp::Neg(_), Type::Float(FloatType::F32)) => Instr::F32_Neg,
            _ => panic!("un-op float {:?} {:?}", op, ty),
        }
    } else if ty == Type::Bool {
        if let syn::UnOp::Not(_) = op {
            Instr::Bool_Not
        } else {
            panic!("un-op bool {:?}", op);
        }
    } else {
        panic!("todo un-op {:?}", ty);
    }*/
}

const SLOT_INVALID: u32 = 0xFFFFFFFF;

impl<'a> BCompiler<'a> {
    fn resolve_loop_jumps(&mut self, loop_id: u32, continue_pc: i32, break_pc: i32) {
        self.loop_jumps.retain(|jump| {
            if jump.loop_id == loop_id {
                if jump.kind == LoopJumpKind::Break {
                    self.code[jump.instr_index] = Instr::Jump(break_pc - jump.instr_index as i32);
                } else {
                    self.code[jump.instr_index] =
                        Instr::Jump(continue_pc - jump.instr_index as i32);
                }
                return false;
            }
            true
        });
    }

    fn get_loop_result(&self, loop_id: u32) -> u32 {
        for entry in &self.loop_results {
            if entry.loop_id == loop_id {
                return entry.slot;
            }
        }
        panic!("no result for loop {}", loop_id);
    }

    fn push_code(&mut self, ins: Instr) {
        assert_eq!(self.code.len(), self.code_frame_depth.len());
        self.code.push(ins);
        self.code_frame_depth.push(self.frame.0);
    }

    fn try_get_place_slot(&mut self, expr_id: u32) -> Option<u32> {
        let expr = &self.input_fn.exprs[expr_id as usize].expr;
        if let Expr::Var(id,_) = expr {
            let res = self.var_map[*id as usize].unwrap();
            Some(res)
        } else if let Expr::IndexTuple(arg, member_n) = expr {
            //let arg_ty = self.input_fn.exprs[*arg as usize].ty;
            let arg_ty = panic!("todo type");
            /*if arg_ty.is_ref() {
                None
            } else if let Some(base) = self.try_get_place_slot(*arg) {
                let layout = arg_ty.layout();
                let offset = layout.member_offsets[*member_n as usize];
                let member_slot = base + offset as u32;

                Some(member_slot)
            } else {
                None
            }*/
        } else if let Some(res) = self.try_get_temp_slot(expr_id) {
            self.lower_expr(expr_id, Some(res));
            Some(res)
        } else {
            None
        }
    }

    // gets the address of something behind an arbitrary number of references
    fn get_struct_addr(&mut self, expr_id: u32) -> u32 {
        panic!("fixme pls");
        /*let ty = panic!("todo type");//&self.input_fn.exprs[expr_id as usize].ty;
        match ty {
            Type::Compound(CompoundType::Ref(wrapped_ty, _)) => {
                let slot = self.lower_expr(expr_id, None);
                if let Type::Compound(CompoundType::Ref(wrapped_ty, _)) = wrapped_ty {
                    let tmp_slot = self.frame.alloc(Type::Int(IntType::USize));

                    self.push_code(Instr::MovSP8(tmp_slot, slot));

                    let mut wrapped_ty = *wrapped_ty;
                    while let Type::Compound(CompoundType::Ref(wt, _)) = wrapped_ty {
                        wrapped_ty = *wt;
                        self.push_code(Instr::MovSP8(tmp_slot, tmp_slot));
                    }

                    tmp_slot
                } else {
                    slot
                }
            }
            _ => self.get_place_addr(expr_id, None),
        }*/
    }

    fn get_place_addr(&mut self, expr_id: u32, mandatory_dest_slot: Option<u32>) -> u32 {
        if let Some(ref_slot) = self.try_get_place_slot(expr_id) {
            // destinations for pointers should always be valid
            let dest_slot =
                mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&GlobalType::simple(TypeKind::ptr())));
            self.push_code(Instr::SlotPtr(dest_slot, ref_slot));

            dest_slot
        } else {
            let expr = &self.input_fn.exprs[expr_id as usize].expr;
            if let Expr::DeRef(arg) = expr {
                self.lower_expr(*arg, mandatory_dest_slot)
            } else if let Expr::IndexTuple(arg, member_n) = expr {
                let arg_ty = &self.input_fn.exprs[*arg as usize].ty;
                let arg_ty = panic!("todo arg ty");
                /*let tuple_ty = arg_ty.get_referenced_r();
                let layout = tuple_ty.layout();
                let offset = layout.member_offsets[*member_n as usize];

                let base_slot = self.get_struct_addr(*arg);
                let dest_slot = mandatory_dest_slot
                    .unwrap_or_else(|| self.frame.alloc(Type::Int(IntType::USize)));
                self.push_code(Instr::OffsetPtr(dest_slot, base_slot, offset));
                dest_slot*/
            } else {
                panic!("todo ref? {:?}", expr)
            }
        }
    }

    fn try_get_temp_slot(&self, expr_id: u32) -> Option<u32> {
        for (id, slot) in &self.temp_slots {
            if *id == expr_id {
                return Some(*slot);
            }
        }
        None
    }

    fn lower_expr(&mut self, expr_id: u32, mandatory_dest_slot: Option<u32>) -> u32 {
        let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[expr_id as usize];

        let ty = ty.clone();

        let saved_frame = self.frame;

        let res_slot = match expr {
            // place exprs
            Expr::Var(..) | Expr::IndexTuple(..) => {
                if let Some(slot) = self.try_get_place_slot(expr_id) {
                    if let Some(mandatory_dest_slot) = mandatory_dest_slot {
                        self.insert_move_ss(mandatory_dest_slot, slot, &ty);
                        mandatory_dest_slot
                    } else {
                        slot
                    }
                } else {
                    let res_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                    let ptr_slot = self.get_place_addr(expr_id, None);
                    self.insert_move_sp(res_slot, ptr_slot, &ty);
                    res_slot
                }
            }

            Expr::DeclVar(id) => {
                assert!(mandatory_dest_slot.is_none());
                let var_info = &self.input_fn.exprs[*id as usize];
                if let Expr::Var(var_index,_) = var_info.expr {
                    let var_ty = var_info.ty.clone();
                    self.var_map[var_index as usize] = Some(self.frame.alloc(&var_ty));
                    SLOT_INVALID
                } else {
                    panic!("bad var decl");
                }
            }
            Expr::DeclTmp(tmp_id) => {
                assert!(mandatory_dest_slot.is_none());
                let tmp_ty = &self.input_fn.exprs[*tmp_id as usize].ty;
                let slot = self.frame.alloc(tmp_ty);
                self.temp_slots.push((*tmp_id, slot));

                SLOT_INVALID
            }
            Expr::StmtTmp(arg, temp_list) => {
                let temp_len = self.temp_slots.len();
                for tmp_id in temp_list {
                    let tmp_ty = &self.input_fn.exprs[*tmp_id as usize].ty;
                    let slot = self.frame.alloc(tmp_ty);
                    //println!("alloc temp {:?}",slot);
                    self.temp_slots.push((*tmp_id, slot));
                }

                self.lower_expr(*arg, None);

                self.temp_slots
                    .resize(temp_len, (SLOT_INVALID, SLOT_INVALID));
                self.frame = saved_frame;
                SLOT_INVALID
            }
            Expr::Assign(dest, src) => {
                // ignore mandatory_dest_slot -- our result is always void

                //let arg_ty = panic!("todo type");

                if let Some(dest_slot) = self.try_get_place_slot(*dest) {
                    self.lower_expr(*src, Some(dest_slot));
                } else {
                    let ptr_slot = self.get_place_addr(*dest, None);
                    let src_slot = self.lower_expr(*src, None);

                    let arg_ty = self.input_fn.exprs[*src as usize].ty.clone();

                    self.insert_move_ps(ptr_slot, src_slot, &arg_ty);
                }
                self.frame = saved_frame; // reset stack
                SLOT_INVALID
            }
            Expr::Ref(arg, _) => self.get_place_addr(*arg, mandatory_dest_slot),
            Expr::DeRef(arg) => {
                // converts a pointer to a value -- this is probably going to cause problems
                let ptr = self.lower_expr(*arg, None);
                self.frame = saved_frame; // arg ready, reset stack
                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                self.insert_move_sp(dest_slot, ptr, &ty);
                dest_slot
            }
            Expr::LitVoid => SLOT_INVALID,
            Expr::LitBool(val) => {
                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                self.push_code(Instr::I8_Const(dest_slot, *val as i8));
                dest_slot
            }
            Expr::LitChar(val) => {
                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                self.push_code(Instr::I32_Const(dest_slot, *val as i32));
                dest_slot
            }
            Expr::LitInt(n,_) => {
                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));

                let width = ty.byte_size();
                match width {
                    1 => self.push_code(Instr::I8_Const(dest_slot, *n as i8)),
                    2 => self.push_code(Instr::I16_Const(dest_slot, *n as i16)),
                    4 => self.push_code(Instr::I32_Const(dest_slot, *n as i32)),
                    8 => self.push_code(Instr::I64_Const(dest_slot, *n as i64)),
                    16 => {
                        let n_ref = Box::new(*n as i128);
                        self.push_code(Instr::I128_Const(dest_slot, n_ref));
                    }
                    _ => panic!("todo more literal ints"),
                };

                dest_slot
            }
            Expr::LitFloat(n,_) => {
                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));

                match ty.kind {
                    TypeKind::Float(Some(FloatWidth::Float64)) => {
                        self.push_code(Instr::F64_Const(dest_slot, *n as f64))
                    }
                    TypeKind::Float(Some(FloatWidth::Float32)) => {
                        self.push_code(Instr::F32_Const(dest_slot, *n as f32))
                    }
                    _ => panic!("todo more literal floats"),
                }

                dest_slot
            }
            Expr::CastPrimitive(src,_) => {
                let src_ty = self.input_fn.exprs[*src as usize].ty.clone();
                let res_ty = ty;

                let src_kind = &src_ty.kind;
                let res_kind = &res_ty.kind;

                let cast_ins_ctor: Option<fn(u32, u32) -> Instr> = if src_kind == res_kind {
                    None
                } else {
                    match (&src_kind,&res_kind) {
                        (TypeKind::Int(_),TypeKind::Int(_)) |
                        (TypeKind::Bool,TypeKind::Int(_)) |
                        (TypeKind::Char,TypeKind::Int(_)) |
                        (TypeKind::Int(Some((IntWidth::Int8,IntSign::Unsigned))),TypeKind::Char) => {
                            let src_width = src_ty.byte_size();
                            let res_width = res_ty.byte_size();

                            if src_width < res_width {
                                let signed = if let TypeKind::Int(Some((_,IntSign::Signed))) = src_kind {
                                    true
                                } else {
                                    false
                                };

                                Some(match (src_width, res_width, signed) {
                                    (1, 2, true) => Instr::I16_S_Widen_8,
                                    (1, 2, false) => Instr::I16_U_Widen_8,
        
                                    (1, 4, true) => Instr::I32_S_Widen_8,
                                    (1, 4, false) => Instr::I32_U_Widen_8,
                                    (2, 4, true) => Instr::I32_S_Widen_16,
                                    (2, 4, false) => Instr::I32_U_Widen_16,
        
                                    (1, 8, true) => Instr::I64_S_Widen_8,
                                    (1, 8, false) => Instr::I64_U_Widen_8,
                                    (2, 8, true) => Instr::I64_S_Widen_16,
                                    (2, 8, false) => Instr::I64_U_Widen_16,
                                    (4, 8, true) => Instr::I64_S_Widen_32,
                                    (4, 8, false) => Instr::I64_U_Widen_32,
        
                                    (1, 16, true) => Instr::I128_S_Widen_8,
                                    (1, 16, false) => Instr::I128_U_Widen_8,
                                    (2, 16, true) => Instr::I128_S_Widen_16,
                                    (2, 16, false) => Instr::I128_U_Widen_16,
                                    (4, 16, true) => Instr::I128_S_Widen_32,
                                    (4, 16, false) => Instr::I128_U_Widen_32,
                                    (8, 16, true) => Instr::I128_S_Widen_64,
                                    (8, 16, false) => Instr::I128_U_Widen_64,
        
                                    _ => panic!("todo widen {} {} {}", src_width, res_width, signed),
                                })
                            } else {
                                None
                            }
                        }
                        // float -> float
                        (TypeKind::Float(Some(FloatWidth::Float32)),TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_F32)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)),TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_F64)
                        }
                        // int -> f32
                        (TypeKind::Int(Some((IntWidth::Int8,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I8_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int8,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I8_U)
                        }
                        (TypeKind::Int(Some((IntWidth::Int16,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I16_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int16,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I16_U)
                        }
                        (TypeKind::Int(Some((IntWidth::Int32,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I32_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int32,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I32_U)
                        }
                        (TypeKind::Int(Some((IntWidth::Int64,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I64_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int64,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I64_U)
                        }
                        (TypeKind::Int(Some((IntWidth::Int128,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I128_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int128,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float32))) => {
                            Some(Instr::F32_From_I128_U)
                        }
                        // int -> f64
                        (TypeKind::Int(Some((IntWidth::Int8,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I8_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int8,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I8_U)
                        }
                        (TypeKind::Int(Some((IntWidth::Int16,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I16_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int16,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I16_U)
                        }
                        (TypeKind::Int(Some((IntWidth::Int32,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I32_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int32,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I32_U)
                        }
                        (TypeKind::Int(Some((IntWidth::Int64,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I64_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int64,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I64_U)
                        }
                        (TypeKind::Int(Some((IntWidth::Int128,IntSign::Signed))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I128_S)
                        }
                        (TypeKind::Int(Some((IntWidth::Int128,IntSign::Unsigned))), TypeKind::Float(Some(FloatWidth::Float64))) => {
                            Some(Instr::F64_From_I128_U)
                        }
                        // f32 -> int
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int8,IntSign::Signed)))) => {
                            Some(Instr::F32_Into_I8_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int8,IntSign::Unsigned)))) => {
                            Some(Instr::F32_Into_I8_U)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int16,IntSign::Signed)))) => {
                            Some(Instr::F32_Into_I16_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int16,IntSign::Unsigned)))) => {
                            Some(Instr::F32_Into_I16_U)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int32,IntSign::Signed)))) => {
                            Some(Instr::F32_Into_I32_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int32,IntSign::Unsigned)))) => {
                            Some(Instr::F32_Into_I32_U)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int64,IntSign::Signed)))) => {
                            Some(Instr::F32_Into_I64_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int64,IntSign::Unsigned)))) => {
                            Some(Instr::F32_Into_I64_U)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int128,IntSign::Signed)))) => {
                            Some(Instr::F32_Into_I128_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float32)), TypeKind::Int(Some((IntWidth::Int128,IntSign::Unsigned)))) => {
                            Some(Instr::F32_Into_I128_U)
                        }
                        // f64 -> int
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int8,IntSign::Signed)))) => {
                            Some(Instr::F64_Into_I8_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int8,IntSign::Unsigned)))) => {
                            Some(Instr::F64_Into_I8_U)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int16,IntSign::Signed)))) => {
                            Some(Instr::F64_Into_I16_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int16,IntSign::Unsigned)))) => {
                            Some(Instr::F64_Into_I16_U)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int32,IntSign::Signed)))) => {
                            Some(Instr::F64_Into_I32_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int32,IntSign::Unsigned)))) => {
                            Some(Instr::F64_Into_I32_U)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int64,IntSign::Signed)))) => {
                            Some(Instr::F64_Into_I64_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int64,IntSign::Unsigned)))) => {
                            Some(Instr::F64_Into_I64_U)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int128,IntSign::Signed)))) => {
                            Some(Instr::F64_Into_I128_S)
                        }
                        (TypeKind::Float(Some(FloatWidth::Float64)), TypeKind::Int(Some((IntWidth::Int128,IntSign::Unsigned)))) => {
                            Some(Instr::F64_Into_I128_U)
                        }
                        _ => {
                            println!(">> {:?} {:?}",src_kind,res_kind);
                            panic!("todo prim cast");
                        }
                    }
                };

                if let Some(cast_ins_ctor) = cast_ins_ctor {
                    let src_slot = self.lower_expr(*src, None);
                    self.frame = saved_frame; // arg ready, reset stack
                    let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&res_ty));
                    let ins = cast_ins_ctor(dest_slot, src_slot);
                    self.push_code(ins);
                    dest_slot
                } else {
                    // no-op cast
                    self.lower_expr(*src, mandatory_dest_slot)
                }
            }
            Expr::BinOp(lhs, op, rhs) => {
                let arg_ty = self.input_fn.exprs[*lhs as usize].ty.clone();

                match op {
                    syn::BinOp::And(_) => {
                        return self.lower_lazy_logic(true, *lhs, *rhs, mandatory_dest_slot);
                    }
                    syn::BinOp::Or(_) => {
                        return self.lower_lazy_logic(false, *lhs, *rhs, mandatory_dest_slot);
                    }
                    _ => (),
                }

                let (ins_ctor, flag) = instr_for_bin_op(*op, &arg_ty);

                if flag == BinOpFlag::Assign {
                    // ignore mandatory_dest_slot -- our result is always void

                    let r_slot = self.lower_expr(*rhs, None);
                    if let Some(l_slot) = self.try_get_place_slot(*lhs) {
                        let ins = ins_ctor(l_slot, l_slot, r_slot);
                        self.push_code(ins);
                    } else {
                        let tmp_slot =self.frame.alloc(&arg_ty);
                        let ptr_slot = self.get_place_addr(*lhs, None);

                        // move value to stack
                        self.insert_move_sp(tmp_slot, ptr_slot, &arg_ty);

                        let ins = ins_ctor(tmp_slot, tmp_slot, r_slot);
                        self.push_code(ins);

                        // move value back to ptr
                        self.insert_move_ps(ptr_slot, tmp_slot, &arg_ty);
                    }
                    self.frame = saved_frame; // all done, reset stack

                    SLOT_INVALID
                } else {
                    let l_slot = self.lower_expr(*lhs, None);
                    let r_slot = self.lower_expr(*rhs, None);

                    self.frame = saved_frame; // args ready, reset stack

                    let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));

                    let ins = if flag == BinOpFlag::SwapArgs {
                        ins_ctor(dest_slot, r_slot, l_slot)
                    } else {
                        ins_ctor(dest_slot, l_slot, r_slot)
                    };
                    self.push_code(ins);
                    dest_slot
                }
            }
            Expr::UnOp(arg, op) => {
                let ins_ctor = instr_for_un_op(*op, &ty);

                let arg_slot = self.lower_expr(*arg, None);
                self.frame = saved_frame; // arg ready, reset stack
                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                let ins = ins_ctor(dest_slot, arg_slot);
                self.push_code(ins);
                dest_slot
            }
            Expr::CallBuiltin(name, args) => {
                if name == "print_int" {
                    let arg_slot = self.lower_expr(args[0], None);
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_int(arg_slot));
                } else if name == "print_uint" {
                    let arg_slot = self.lower_expr(args[0], None);
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_uint(arg_slot));
                } else if name == "print_float" {
                    let arg_slot = self.lower_expr(args[0], None);
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_float(arg_slot));
                } else if name == "print_bool" {
                    let arg_slot = self.lower_expr(args[0], None);
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_bool(arg_slot));
                } else if name == "print_char" {
                    let arg_slot = self.lower_expr(args[0], None);
                    self.frame = saved_frame; // args ready, reset stack
                    self.push_code(Instr::BuiltIn_print_char(arg_slot));
                } else {
                    panic!("unknown builtin");
                }
                SLOT_INVALID
            }
            Expr::Block(block) => self.lower_block(block, &ty, mandatory_dest_slot),
            Expr::Break(loop_id, break_expr) => {
                if let Some(break_expr) = break_expr {
                    let dest_slot = self.get_loop_result(*loop_id);
                    self.lower_expr(*break_expr, Some(dest_slot));
                }
                let instr_index = self.code.len();
                self.push_code(Instr::Bad);
                self.loop_jumps.push(LoopJump {
                    kind: LoopJumpKind::Break,
                    loop_id: *loop_id,
                    instr_index,
                });
                SLOT_INVALID
            }
            Expr::Continue(loop_id) => {
                let instr_index = self.code.len();
                self.push_code(Instr::Bad);
                self.loop_jumps.push(LoopJump {
                    kind: LoopJumpKind::Continue,
                    loop_id: *loop_id,
                    instr_index,
                });
                SLOT_INVALID
            }
            Expr::While(cond, body) => {
                // ignore mandatory_dest_slot
                let pc_start = self.code.len() as i32;
                let cond_slot = self.frame.alloc(&GlobalType::simple(TypeKind::Bool));
                self.lower_expr(*cond, Some(cond_slot));
                let pc_cond_jump = self.code.len();
                self.frame = saved_frame; // cond_read, reset stack
                self.push_code(Instr::Bad);

                self.lower_block(body, &GlobalType::simple(TypeKind::Tuple), None);

                let pc_end = self.code.len() as i32;
                self.push_code(Instr::Jump(pc_start - pc_end));
                // need to offset + 1 to make it past the end
                self.code[pc_cond_jump] = Instr::JumpF(pc_end - pc_cond_jump as i32 + 1, cond_slot);

                self.resolve_loop_jumps(expr_id, pc_start, pc_end + 1);

                SLOT_INVALID
            }
            Expr::Loop(body) => {
                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                self.loop_results.push(LoopResult {
                    loop_id: expr_id,
                    slot: dest_slot,
                });

                let pc_start = self.code.len() as i32;
                self.lower_block(body, &GlobalType::simple(TypeKind::Tuple), None);
                let pc_end = self.code.len() as i32;
                self.push_code(Instr::Jump(pc_start - pc_end));

                self.loop_results.pop();
                self.resolve_loop_jumps(expr_id, pc_start, pc_end + 1);

                dest_slot
            }
            Expr::If(cond, then_block, else_expr) => {
                let cond_slot = self.frame.alloc(&GlobalType::simple(TypeKind::Bool));
                self.lower_expr(*cond, Some(cond_slot));
                let pc_jump_then = self.code.len();
                self.frame = saved_frame; // cond_read, reset stack
                self.push_code(Instr::Bad);

                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                let saved_frame = self.frame;

                self.lower_block(then_block, &ty, Some(dest_slot));

                if let Some(else_expr) = else_expr {
                    let pc_jump_else = self.code.len();
                    self.push_code(Instr::Bad);

                    let pc_end_then = self.code.len() as i32;

                    self.lower_expr(*else_expr, Some(dest_slot));

                    let pc_end_else = self.code.len() as i32;

                    self.code[pc_jump_then] =
                        Instr::JumpF(pc_end_then - pc_jump_then as i32, cond_slot);
                    self.code[pc_jump_else] = Instr::Jump(pc_end_else - pc_jump_else as i32);
                } else {
                    let pc_end_then = self.code.len() as i32;
                    self.code[pc_jump_then] =
                        Instr::JumpF(pc_end_then - pc_jump_then as i32, cond_slot);
                }

                self.frame = saved_frame;
                dest_slot
            }
            Expr::Call(func, args) => {
                let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                let saved_frame = self.frame;

                let call_base = self.frame.align_for_call();
                let sig = func.sig();
                let ret_ptr_slot = self.frame.alloc(&GlobalType::simple(TypeKind::ptr()));
                assert_eq!(call_base, ret_ptr_slot);

                // the return pointer slot should point to the result slot
                self.push_code(Instr::SlotPtr(ret_ptr_slot, dest_slot));

                for (ty, ex) in sig.inputs.iter().zip(args.iter()) {
                    let slot = self.frame.alloc(ty);

                    let arg_frame = self.frame;
                    self.lower_expr(*ex, Some(slot));
                    assert!(arg_frame == self.frame);
                }

                self.push_code(Instr::Call(call_base, func));

                self.frame = saved_frame;
                dest_slot
            }
            Expr::NewTuple(args) => {
                panic!("tuple")
                /*let layout = ty.layout();

                let base_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&ty));
                let saved_frame = self.frame;

                for (arg_id, offset) in args.iter().zip(layout.member_offsets.iter()) {
                    let member_slot = base_slot + *offset as u32;
                    self.lower_expr(*arg_id, Some(member_slot));
                }

                self.frame = saved_frame;
                base_slot*/
            }
            Expr::Return(arg) => {
                let slot = if let Some(arg) = arg {
                    self.lower_expr(*arg, None)
                } else {
                    SLOT_INVALID
                };
                self.insert_return(slot);

                SLOT_INVALID
            }
            _ => panic!("vm compile {:?}", expr),
        };
        // NOTE: Never results can yield None even with a mandatory result.
        if mandatory_dest_slot.is_some() && res_slot != SLOT_INVALID {
            assert_eq!(mandatory_dest_slot.unwrap(), res_slot);
        }

        res_slot
    }

    fn lower_lazy_logic(
        &mut self,
        is_and: bool,
        lhs: u32,
        rhs: u32,
        mandatory_dest_slot: Option<u32>,
    ) -> u32 {
        let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(&GlobalType::simple(TypeKind::Bool)));
        let saved_frame = self.frame;

        self.lower_expr(lhs, Some(dest_slot));
        self.frame = saved_frame; // reset stack
        let pc_jump = self.code.len();
        self.push_code(Instr::Bad);

        self.lower_expr(rhs, Some(dest_slot));
        self.frame = saved_frame; // reset stack
        let pc_end = self.code.len() as i32;

        if is_and {
            self.code[pc_jump] = Instr::JumpF(pc_end - pc_jump as i32, dest_slot);
        } else {
            self.code[pc_jump] = Instr::JumpT(pc_end - pc_jump as i32, dest_slot);
        }

        dest_slot
    }

    fn lower_block(&mut self, block: &Block, ty: &GlobalType, mandatory_dest_slot: Option<u32>) -> u32 {
        let dest_slot = mandatory_dest_slot.unwrap_or_else(|| self.frame.alloc(ty));
        let saved_frame = self.frame;
        let temp_len = self.temp_slots.len();

        for expr_id in &block.stmts {
            self.lower_expr(*expr_id, None);
        }

        if let Some(expr_id) = &block.result {
            self.lower_expr(*expr_id, Some(dest_slot));
        }
        self.frame = saved_frame;
        self.temp_slots
            .resize(temp_len, (SLOT_INVALID, SLOT_INVALID));
        dest_slot
    }

    fn insert_move_ss(&mut self, dest: u32, src: u32, ty: &GlobalType) {
        let size = ty.byte_size();
        if size == 0 {
            return;
        }

        let align = ty.byte_align();

        match align {
            1 => {
                if size == 1 {
                    self.push_code(Instr::MovSS1(dest, src));
                } else {
                    self.push_code(Instr::MovSS1N(dest, src, size as u32));
                }
            }
            2 => {
                if size == 2 {
                    self.push_code(Instr::MovSS2(dest, src));
                } else {
                    assert_eq!(size % 2, 0);
                    self.push_code(Instr::MovSS2N(dest, src, size as u32 / 2));
                }
            }
            4 => {
                if size == 4 {
                    self.push_code(Instr::MovSS4(dest, src));
                } else {
                    assert_eq!(size % 4, 0);
                    self.push_code(Instr::MovSS4N(dest, src, size as u32 / 4));
                }
            }
            8 => {
                if size == 8 {
                    self.push_code(Instr::MovSS8(dest, src));
                } else {
                    assert_eq!(size % 8, 0);
                    self.push_code(Instr::MovSS8N(dest, src, size as u32 / 8));
                }
            }
            16 => {
                if size == 16 {
                    self.push_code(Instr::MovSS16(dest, src));
                } else {
                    assert_eq!(size % 16, 0);
                    self.push_code(Instr::MovSS16N(dest, src, size as u32 / 16));
                }
            }
            _ => panic!("no move ss {} {}", size, align),
        }
    }

    fn insert_move_ps(&mut self, dest: u32, src: u32, ty: &GlobalType) {
        let size = ty.byte_size();
        if size == 0 {
            return;
        }

        let align = ty.byte_align();

        if size == 1 && align == 1 {
            self.push_code(Instr::MovPS1(dest, src));
        } else if size == 2 && align == 2 {
            self.push_code(Instr::MovPS2(dest, src));
        } else if size == 4 && align == 4 {
            self.push_code(Instr::MovPS4(dest, src));
        } else if size == 8 && align == 8 {
            self.push_code(Instr::MovPS8(dest, src));
        } else if size == 16 && align == 16 {
            self.push_code(Instr::MovPS16(dest, src));
        } else {
            panic!("no move ps {} {}", size, align);
        }
    }

    fn insert_move_sp(&mut self, dest: u32, src: u32, ty: &GlobalType) {
        // todo ignore type alignment and use src/dst alignment
        let size = ty.byte_size();
        if size == 0 {
            return;
        }

        let align = ty.byte_align();

        if size == 1 && align == 1 {
            self.push_code(Instr::MovSP1(dest, src));
        } else if size == 2 && align == 2 {
            self.push_code(Instr::MovSP2(dest, src));
        } else if size == 4 && align == 4 {
            self.push_code(Instr::MovSP4(dest, src));
        } else if size == 8 && align == 8 {
            self.push_code(Instr::MovSP8(dest, src));
        } else if size == 16 && align == 16 {
            self.push_code(Instr::MovSP16(dest, src));
        } else {
            panic!("no move sp {} {}", size, align);
        }
    }

    fn insert_return(&mut self, res: u32) {
        //let ty = self.input_fn.exprs[self.input_fn.root_expr].ty;
        let ty = self.input_fn.ret_ty.clone();
        self.insert_move_ps(0, res, &ty);
        self.push_code(Instr::Return);
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
