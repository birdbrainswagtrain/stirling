use std::cell::Cell;

use syn::{BinOp, UnOp};

use crate::builtin::BUILTINS;

use super::{func::{Expr, FuncHIR}, types::global::GlobalType};
use super::types::common::{TypeKind, IntWidth, IntSign, FloatWidth};

#[derive(Debug,Clone,Copy,PartialEq)]
pub struct TypeVar(u32);

enum UnifyResult {
    A,
    B
}

fn unify_kind(a: &TypeKind, b: &TypeKind) -> UnifyResult {
    if a == b {
        return UnifyResult::A;
    }

    match (a,b) {
        (_,TypeKind::Unknown) => UnifyResult::A,
        (TypeKind::Unknown,_) => UnifyResult::B,

        (TypeKind::Int(Some(_)),TypeKind::Int(None)) => UnifyResult::A,
        (TypeKind::Int(None),TypeKind::Int(Some(_))) => UnifyResult::B,

        (TypeKind::Float(Some(_)),TypeKind::Float(None)) => UnifyResult::A,
        (TypeKind::Float(None),TypeKind::Float(Some(_))) => UnifyResult::B,

        _ => panic!("todo unify kinds {:?} {:?}",a,b)
    }
}

#[derive(Debug,Clone,PartialEq)]
struct LocalType {
    kind: TypeKind,
    args: Vec<TypeVar>
}

#[derive(Debug)]
enum TypeConstraint {
    Equal(TypeVar,TypeVar),
    CoerceTo(TypeVar,GlobalType),
    Assign{src: TypeVar, dst: TypeVar},
    DeRef{src: TypeVar, dst: TypeVar},
    OpBinary{op: OpBinary, res: TypeVar, lhs: TypeVar, rhs: TypeVar},
    OpUnary{op: OpUnary, res: TypeVar, arg: TypeVar},
    CheckBlockNever{var: TypeVar, expr_id: u32},
    LeastUpperBound2{res: TypeVar, arg1: TypeVar, arg2: TypeVar}
}

impl TypeConstraint {
    // used for conditions and &&/|| args.
    // todo possibly allow never?
    fn bool(var: TypeVar) -> Self {
        TypeConstraint::CoerceTo(var,GlobalType::simple(TypeKind::Bool))
    }
    // todo same for void?
}

#[derive(Debug,Clone,Copy)]
enum OpUnary {
    Neg,
    Not
}

#[derive(Debug,Clone,Copy)]
enum OpBinary {
    Ord,
    Eq,

    Add,
    Sub,
    Mul,
    Div,
    Rem,

    BitOr,
    BitAnd,
    BitXor,

    ShiftL,
    ShiftR,

    AddEq,
    SubEq,
    MulEq,
    DivEq,
    RemEq,

    BitOrEq,
    BitAndEq,
    BitXorEq,

    ShiftLEq,
    ShiftREq,
}

fn op_constraint_and_type(op: &BinOp) -> (OpBinary,TypeKind) {
    match op {
        BinOp::Lt(_) | BinOp::Gt(_) | BinOp::Le(_) | BinOp::Ge(_) => (OpBinary::Ord,TypeKind::Bool),

        BinOp::Eq(_) | BinOp::Ne(_) => (OpBinary::Eq,TypeKind::Bool),

        BinOp::Add(_) => (OpBinary::Add,TypeKind::Unknown),
        BinOp::Sub(_) => (OpBinary::Sub,TypeKind::Unknown),
        BinOp::Mul(_) => (OpBinary::Mul,TypeKind::Unknown),
        BinOp::Div(_) => (OpBinary::Div,TypeKind::Unknown),
        BinOp::Rem(_) => (OpBinary::Rem,TypeKind::Unknown),

        BinOp::BitOr(_) => (OpBinary::BitOr,TypeKind::Unknown),
        BinOp::BitAnd(_) => (OpBinary::BitAnd,TypeKind::Unknown),
        BinOp::BitXor(_) => (OpBinary::BitXor,TypeKind::Unknown),

        BinOp::Shl(_) => (OpBinary::ShiftL,TypeKind::Unknown),
        BinOp::Shr(_) => (OpBinary::ShiftR,TypeKind::Unknown),

        // remaining ops result in void
        BinOp::AddEq(_) => (OpBinary::AddEq,TypeKind::Tuple),
        BinOp::SubEq(_) => (OpBinary::SubEq,TypeKind::Tuple),
        BinOp::MulEq(_) => (OpBinary::MulEq,TypeKind::Tuple),
        BinOp::DivEq(_) => (OpBinary::DivEq,TypeKind::Tuple),
        BinOp::RemEq(_) => (OpBinary::RemEq,TypeKind::Tuple),

        BinOp::BitOrEq(_) => (OpBinary::BitOrEq,TypeKind::Tuple),
        BinOp::BitAndEq(_) => (OpBinary::BitAndEq,TypeKind::Tuple),
        BinOp::BitXorEq(_) => (OpBinary::BitXorEq,TypeKind::Tuple),

        BinOp::ShlEq(_) => (OpBinary::ShiftLEq,TypeKind::Tuple),
        BinOp::ShrEq(_) => (OpBinary::ShiftREq,TypeKind::Tuple),

        _ => panic!("todo op {:?}",op)
    }
}

#[derive(Debug)]
struct LoopType{
    expr_id: u32,
    type_var: TypeVar,
    breaks: Cell<bool>
}

pub struct FuncTypes {
    var_types: Vec<LocalType>,
    expr_vars: Vec<TypeVar>,
    constraints: Vec<TypeConstraint>,
    constraints_sat: Vec<TypeConstraint>,
    loop_types: Vec<LoopType>
}

const TYPE_INVALID: TypeVar = TypeVar(0xFFFFFFFF);
const TYPE_VOID: TypeVar = TypeVar(0);

impl FuncTypes {
    pub fn new(func: &mut FuncHIR, arg_count: usize) -> Self {
        let mut types = FuncTypes{
            var_types: vec!(
                LocalType{kind: TypeKind::Tuple, args: vec!()}, // TYPE_VOID
            ),
            expr_vars: vec!(TYPE_INVALID; func.exprs.len()),
            constraints: vec!(),
            constraints_sat: vec!(),
            loop_types: vec!()
        };

        for i in 0..arg_count {
            let var_index = func.vars[i];
            types.init_expr(func, var_index);
        }

        let main_var = types.init_expr(func, func.root_expr as u32);
        types.add_constraint(TypeConstraint::CoerceTo(main_var, func.ret_ty.clone()));

        types
    }

    pub fn init_expr(&mut self, func: &FuncHIR, expr_id: u32) -> TypeVar {
        {
            let old = self.expr_vars[expr_id as usize];
            if old != TYPE_INVALID {
                return old;
            }
        }

        let expr = &func.exprs[expr_id as usize].expr;
        let result = match expr {
            Expr::StmtTmp(arg,_) => {
                self.init_expr(func,*arg)
            }
            Expr::DeclTmp(_) => {
                // no-op
                TYPE_VOID
            }
            Expr::DeclVar(_) => {
                // no-op
                TYPE_VOID
            }
            Expr::Assign(dst, src) => {
                let dst = self.init_expr(func,*dst);
                let src = self.init_expr(func,*src);
                
                self.add_constraint(TypeConstraint::Assign{src,dst});

                TYPE_VOID
            }
            Expr::BinOp(lhs, op, rhs) => {
                let lhs = self.init_expr(func,*lhs);
                let rhs = self.init_expr(func,*rhs);

                match op {
                    BinOp::And(_) | BinOp::Or(_) => {
                        self.add_constraint(TypeConstraint::bool(lhs));
                        self.add_constraint(TypeConstraint::bool(rhs));
                        self.new_var(TypeKind::Bool)
                    },
                    _ => {
                        let (op,ty) = op_constraint_and_type(op);
                        let res = self.new_var(ty);
        
                        self.add_constraint(TypeConstraint::OpBinary{op, res, lhs, rhs});
        
                        res
                    }
                }
            }
            Expr::UnOp(arg, op) => {
                let arg = self.init_expr(func,*arg);
                let res = self.new_var(TypeKind::Unknown);
                match op {
                    UnOp::Not(_) => self.add_constraint(TypeConstraint::OpUnary{op:OpUnary::Not, res,arg}),
                    UnOp::Neg(_) => self.add_constraint(TypeConstraint::OpUnary{op:OpUnary::Neg, res,arg}),
                    UnOp::Deref(_) => panic!()
                }
                res
            }
            Expr::Var(_,ty) => {
                self.new_var_from_global(ty)
            }
            Expr::CastPrimitive(arg, ty) => {
                self.init_expr(func,*arg);

                self.new_var_from_global(ty)
            }
            Expr::LitInt(_,ty) => {
                self.new_var(TypeKind::Int(*ty))
            }
            Expr::LitFloat(_,ty) => {
                self.new_var(TypeKind::Float(*ty))
            }
            Expr::LitBool(_) => {
                self.new_var(TypeKind::Bool)
            }
            Expr::LitChar(_) => {
                self.new_var(TypeKind::Char)
            }
            Expr::LitVoid => {
                self.new_var(TypeKind::Tuple)
            }
            Expr::Block(block) => {
                
                for stmt in &block.stmts {
                    self.init_expr(func,*stmt);
                }
                
                if let Some(res) = block.result {
                    self.init_expr(func,res)
                    //self.add_constraint(TypeConstraint::Equal(block_var, res_var));
                } else {
                    let block_var = self.new_var(TypeKind::Unknown);
                    self.add_constraint(TypeConstraint::CheckBlockNever{ var: block_var, expr_id });
                    block_var
                }
            }
            Expr::If(cond, then_block, else_expr) => {
                let cond_var = self.init_expr(func,*cond);
                self.add_constraint(TypeConstraint::bool(cond_var));

                for stmt in &then_block.stmts {
                    self.init_expr(func,*stmt);
                }

                if let Some(else_expr) = else_expr {
                    let then_var = if let Some(res) = then_block.result {
                        self.init_expr(func, res)
                    } else {
                        let then_var = self.new_var(TypeKind::Unknown);
                        self.add_constraint(TypeConstraint::CheckBlockNever{ var: then_var, expr_id });
                        then_var
                    };

                    let else_var = self.init_expr(func,*else_expr);

                    let res_var = self.new_var(TypeKind::Unknown);

                    self.add_constraint(TypeConstraint::LeastUpperBound2{ res: res_var, arg1: then_var, arg2: else_var });

                    res_var
                } else {
                    if let Some(res) = then_block.result {
                        let res_var = self.init_expr(func,res);
                        self.add_constraint(TypeConstraint::Equal(res_var,TYPE_VOID));
                    }
                    TYPE_VOID
                }
            }
            Expr::While(cond, block) => {
                let cond_var = self.init_expr(func,*cond);
                self.add_constraint(TypeConstraint::bool(cond_var));

                for stmt in &block.stmts {
                    self.init_expr(func,*stmt);
                }

                if let Some(res) = block.result {
                    let res_var = self.init_expr(func,res);
                    self.add_constraint(TypeConstraint::Equal(res_var,TYPE_VOID));
                }

                TYPE_VOID
            }
            Expr::Loop(block) => {
                let res = self.new_var(TypeKind::Unknown);

                self.loop_types.push(LoopType { expr_id, type_var: res, breaks: Cell::new(false) });

                // do loop body
                {                    
                    for stmt in &block.stmts {
                        self.init_expr(func,*stmt);
                    }
    
                    if let Some(res) = block.result {
                        let res_var = self.init_expr(func,res);
                        self.add_constraint(TypeConstraint::Equal(res_var,TYPE_VOID));
                    }
                }

                let lt = self.loop_types.pop().unwrap();
                assert_eq!(lt.expr_id,expr_id);
                if !lt.breaks.get() {
                    let ty = &mut self.var_types[res.0 as usize];
                    assert_eq!(ty.kind,TypeKind::Unknown);
                    ty.kind = TypeKind::Never;
                }

                res
            }
            Expr::Break(loop_expr, break_expr) => {
                let mut loop_type = None;
                for lt in &self.loop_types {
                    if lt.expr_id == *loop_expr {
                        loop_type = Some(lt);
                        break;
                    }
                }
                
                if let Some(loop_type) = loop_type {
                    loop_type.breaks.set(true);
                    let dst = loop_type.type_var;
                    if let Some(break_expr) = break_expr {
                        let break_var = self.init_expr(func,*break_expr);
                        self.add_constraint(TypeConstraint::Assign{src: break_var, dst});
                    } else {
                        self.add_constraint(TypeConstraint::Assign{src: TYPE_VOID, dst});
                    }
                } else {
                    assert!(break_expr.is_none());
                }

                self.new_var(TypeKind::Never)
            }
            Expr::Continue(_) => {
                self.new_var(TypeKind::Never)
            }

            Expr::CallBuiltin(name, args) => {
                let entry = BUILTINS.get(name.as_str());
                if entry.is_none() {
                    panic!("invalid builtin: {}", name);
                }

                let sig = &entry.unwrap().1;

                assert_eq!(args.len(),sig.inputs.len());

                for (arg,ty) in args.iter().zip(sig.inputs.iter()) {
                    let var = self.init_expr(func,*arg);
                    self.add_constraint(TypeConstraint::CoerceTo(var,ty.clone()));
                }

                self.new_var_from_global(&sig.output)
            }
            Expr::Call(call_func, args) => {
                let sig = call_func.sig();

                assert_eq!(args.len(),sig.inputs.len());

                for (arg,ty) in args.iter().zip(sig.inputs.iter()) {
                    let var = self.init_expr(func,*arg);
                    self.add_constraint(TypeConstraint::CoerceTo(var,ty.clone()));
                }

                self.new_var_from_global(&sig.output)
            }
            Expr::Return(ret_expr) => {
                let ret_var = if let Some(ret_expr) = ret_expr {
                    self.init_expr(func,*ret_expr)
                } else {
                    TYPE_VOID
                };

                // todo coerce?
                self.add_constraint(TypeConstraint::CoerceTo(ret_var,func.ret_ty.clone()));

                self.new_var(TypeKind::Never)
            }
            Expr::Ref(ref_expr, is_mut) => {
                let ref_ty = self.init_expr(func,*ref_expr);

                self.new_var_with_args(TypeKind::Ref(*is_mut),vec!(ref_ty))
            }
            Expr::DeRef(ref_expr) => {
                let src = self.init_expr(func,*ref_expr);
                let dst = self.new_var(TypeKind::Unknown);
                self.add_constraint(TypeConstraint::DeRef{src,dst});
                dst
            }
            _ => {
                println!("todo setup types: {:?}",expr);
                panic!();
            }
        };

        self.expr_vars[expr_id as usize] = result;

        result
    }

    fn new_var(&mut self, kind: TypeKind) -> TypeVar {
        let res = TypeVar(self.var_types.len() as u32);
        self.var_types.push(LocalType { kind, args: vec!() });
        res
    }

    fn new_var_with_args(&mut self, kind: TypeKind, args: Vec<TypeVar>) -> TypeVar {
        let res = TypeVar(self.var_types.len() as u32);
        self.var_types.push(LocalType { kind, args });
        res
    }

    fn new_var_from_global(&mut self, source: &GlobalType) -> TypeVar {
        let args: Vec<TypeVar> = source.args.iter().map(|arg| {
            self.new_var_from_global(arg)
        }).collect();

        let res = TypeVar(self.var_types.len() as u32);
        self.var_types.push(LocalType { kind: source.kind.clone(), args });

        res
    }

    fn add_constraint(&mut self, c: TypeConstraint) {
        self.constraints.push(c);
    }

    pub fn solve(&mut self, func: &mut FuncHIR) {
        let constraints = std::mem::take(&mut self.constraints);
        for c in constraints {
            let sat = match &c {
                TypeConstraint::Equal(var1, var2) => self.solve_equal(*var1,*var2),
                TypeConstraint::CoerceTo(var,ty) => self.solve_coerce_to(*var,&ty.clone()),
                TypeConstraint::Assign { src, dst } => self.solve_assign(*src,*dst),
                TypeConstraint::DeRef { src, dst } => {
                    // todo std ops deref / derefmut
                    if self.check_known_strict(*src) {
                        if let TypeKind::Ref(_) = self.get(*src).kind {
                            let ref_var = self.get(*src).args[0];
                            self.solve_equal(ref_var, *dst)
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                TypeConstraint::OpUnary{ op, res, arg } => self.solve_unary(*op,*res,*arg),
                TypeConstraint::OpBinary { op, res, lhs, rhs } => self.solve_binary(*op,*res,*lhs,*rhs),
                TypeConstraint::CheckBlockNever{ var, expr_id } => self.solve_block_never(func, *var, *expr_id),
                TypeConstraint::LeastUpperBound2{ res, arg1, arg2 } => self.solve_lub2(*res, *arg1, *arg2),
            };
            if sat {
                self.constraints_sat.push(c);
            } else {
                self.constraints.push(c);
            }
        }
    }

    pub fn solve_desperate(&mut self, func: &mut FuncHIR) {
        for c in &self.constraints {
            match c {
                TypeConstraint::Assign { src, dst } => {
                    let src_ty = self.get(*src);
                    let dst_ty = self.get(*dst);
                    if src_ty.kind == TypeKind::Never && dst_ty.kind == TypeKind::Unknown {
                        self.var_types[dst.0 as usize].kind = TypeKind::Never;
                        return;
                    }
                }
                TypeConstraint::CoerceTo( a, b) => {
                    self.solve_equal_g(*a,&b.clone());
                    return;
                }
                _ => ()
            }
        }
    }

    fn get(&self, var: TypeVar) -> &LocalType {
        let var = var.0 as usize;
        &self.var_types[var]
    }

    fn get_mut_2(&mut self, var1: TypeVar, var2: TypeVar) -> (&mut LocalType, &mut LocalType) {
        let var1 = var1.0 as isize;
        let var2 = var2.0 as isize;

        // safety: indices must be distinct and valid
        assert!(var1 != var2);
        assert!((var1 as usize) < self.var_types.len());
        assert!((var2 as usize) < self.var_types.len());

        let ptr = self.var_types.as_mut_ptr();
        unsafe {
            (&mut *ptr.offset(var1 as isize),&mut *ptr.offset(var2 as isize))
        }
    }

    fn get_global(&self, var: TypeVar) -> GlobalType {
        let var = var.0 as usize;
        let ty = &self.var_types[var];
        let args: Vec<GlobalType> = ty.args.iter().map(|arg_var| {
            self.get_global(*arg_var)
        }).collect();
        GlobalType::with_args(ty.kind.clone(),&args)
    }

    fn solve_equal(&mut self, var1: TypeVar, var2: TypeVar) -> bool {
        if var1 == var2 {
            return true;
            //return self.get(var1).kind.is_known_strict();
        }

        let (a,b) = self.get_mut_2(var1, var2);

        let unify_res = unify_kind(&a.kind, &b.kind);

        let (src,dst) = match unify_res {
            UnifyResult::A => {
                (a,b)
            }
            UnifyResult::B => {
                (b,a)
            }
        };

        let dst_is_unk = dst.kind == TypeKind::Unknown;
        if dst_is_unk {
            dst.args = src.args.clone();
        }

        dst.kind = src.kind.clone();

        if src.args.len() > 0 && !dst_is_unk {
            assert_eq!(src.args.len(),dst.args.len());
            // yucky clones
            let args1 = src.args.clone();
            let args2 = dst.args.clone();
            for (a1,a2) in args1.iter().zip(args2.iter()) {
                self.solve_equal(*a1, *a2);
            }
            // use only one set of arguments
            {
                let (a,b) = self.get_mut_2(var1, var2);
                a.args = b.args.clone();
            }
        }

        self.check_known_strict(var1)
    }

    fn solve_coerce_to(&mut self, var: TypeVar, ty: &GlobalType) -> bool {
        let src = &mut self.var_types[var.0 as usize];

        if src.kind == TypeKind::Never {
            // never can coerce to anything
            true
        } else if src.kind.cannot_coerce() {
            // source cannot be coerced, it must equal ty
            self.solve_equal_g(var,ty)
        } else {
            // not sure what to do
            false
        }
    }

    fn solve_equal_g(&mut self, var: TypeVar, ty: &GlobalType) -> bool {
        let src = &mut self.var_types[var.0 as usize];

        let unify_res = unify_kind(&src.kind, &ty.kind);
        match unify_res {
            UnifyResult::A => (),
            UnifyResult::B => src.kind = ty.kind.clone()
        }

        assert_eq!(src.args.len(),ty.args.len());
        assert_eq!(ty.args.len(),0);

        self.check_known_strict(var)
    }

    fn solve_assign(&mut self, src_var: TypeVar, dst_var: TypeVar) -> bool {
        // todo write more robust coercion logic that can be re-used elsewhere
        let src = self.get(src_var);
        let dst = self.get(dst_var);
        if dst.kind == TypeKind::Unknown {
            if src.kind != TypeKind::Never {
                return self.solve_equal(src_var, dst_var);
            }
        } else {
            // src must coerce to dst
            if src.kind == TypeKind::Never {
                // src is never and dst is known -- this is a valid coercion
                return true;
            } else if src.kind != TypeKind::Unknown {
                // src and dst are both known, make them equal
                return self.solve_equal(src_var, dst_var);
            }
        }

        self.get(src_var).kind.is_known_strict() && self.get(dst_var).kind.is_known_strict()
    }

    fn solve_lub2(&mut self, res_var: TypeVar, arg1_var: TypeVar, arg2_var: TypeVar) -> bool {

        // if either argument is of a known type that cannot coerce to another type, assume it is our result type
        if self.get(arg1_var).kind.cannot_coerce() {
            self.solve_equal(arg1_var, res_var);
            return self.get(arg1_var).kind.is_known_strict() && self.get(arg2_var).kind.is_known_strict();
        }
        if self.get(arg2_var).kind.cannot_coerce() {
            self.solve_equal(arg2_var, res_var);
            return self.get(arg1_var).kind.is_known_strict() && self.get(arg2_var).kind.is_known_strict();
        }

        // big, massive todo
        if self.get(arg1_var) == self.get(arg2_var) {
            self.solve_equal(arg1_var, res_var)
        } else {
            false
        }
    }

    fn solve_unary(&mut self, _op: OpUnary, res: TypeVar, arg: TypeVar) -> bool {
        if self.get(arg).kind.is_op_prim() {
            self.solve_equal(arg, res)
        } else {
            // TODO TRAIT?
            false
        }
    }

    fn solve_binary(&mut self, op: OpBinary, res: TypeVar, lhs: TypeVar, rhs: TypeVar) -> bool {
        match op {
            OpBinary::ShiftLEq | OpBinary::ShiftREq => {
                if self.get(lhs).kind.is_op_prim() && self.get(rhs).kind.is_op_prim() {
                    // not much we can do
                    true
                } else {
                    // todo trait lookup
                    false
                }
            }
            OpBinary::Eq | OpBinary::Ord |
            OpBinary::AddEq | OpBinary::SubEq | OpBinary::MulEq | OpBinary::DivEq | OpBinary::RemEq |
            OpBinary::BitAndEq | OpBinary::BitOrEq | OpBinary::BitXorEq => {
                if self.get(lhs).kind.is_op_prim() && self.get(rhs).kind.is_op_prim() {
                    self.solve_equal(lhs, rhs)
                } else {
                    // todo trait lookup
                    false
                }
            }
            OpBinary::ShiftL | OpBinary::ShiftR => {

                if self.get(lhs).kind.is_op_prim() && self.get(rhs).kind.is_op_prim() {
                    // lhs and res should match
                    self.solve_equal(lhs, res)
                } else {
                    // todo trait lookup
                    false
                }
            }
            OpBinary::Add | OpBinary::Sub | OpBinary::Mul | OpBinary::Div | OpBinary::Rem |
            OpBinary::BitAnd | OpBinary::BitOr | OpBinary::BitXor => {

                if self.get(lhs).kind.is_op_prim() && self.get(rhs).kind.is_op_prim() {
                    self.solve_equal(lhs, rhs);
                    self.solve_equal(rhs, res);
                    self.solve_equal(res, lhs);

                    self.get(lhs).kind.is_known_strict()
                } else {
                    // todo trait lookup
                    false
                }
            }
            _ => panic!("? {:?}",op)
        }
    }

    fn solve_block_never(&mut self, func: &FuncHIR, var: TypeVar, expr_id: u32) -> bool {
        // check type already resolved
        let ty = self.get(var);
        if ty.kind.is_known_strict() {
            return true;
        } 
        
        let expr = &func.exprs[expr_id as usize].expr;
        let block = match expr {
            Expr::Block(block) => {
                block
            }
            Expr::If(_,block,_) => {
                block
            }
            _ => panic!("todo never check root {:?}",expr)
        };
        let mut is_never = false;
        for stmt_id in &block.stmts {
            let res = self.check_expr_is_never(func,*stmt_id);
            match res {
                None => return false, // can't determine type, bail out
                Some(true) => {
                    is_never = true;
                    break;
                }
                Some(false) => ()
            }
        }
        let goal_kind = if is_never { TypeKind::Never } else { TypeKind::Tuple };

        let current_ty = &mut self.var_types[var.0 as usize];

        if current_ty.kind != TypeKind::Unknown {
            panic!("block already has type");
        }
        current_ty.kind = goal_kind;

        return true;
    }

    fn check_expr_is_never(&self, func: &FuncHIR, expr_id: u32) -> Option<bool> {
        let expr = &func.exprs[expr_id as usize].expr;
        match expr {

            // check children
            Expr::UnOp(arg,_) |
            Expr::Ref(arg,_) |
            Expr::DeRef(arg) => {
                self.check_expr_is_never(func, *arg)
            }
            Expr::Assign(a,b) |
            Expr::BinOp(a,_,b) => {
                let a = self.check_expr_is_never(func, *a)?;
                let b = self.check_expr_is_never(func, *b)?;
                Some(a || b)
            }
            // check args and type of the expression
            Expr::Call(_,args) |
            Expr::CallBuiltin(_,args) => {
                // assume builtins never diverge
                for arg in args {
                    let arg = self.check_expr_is_never(func, *arg)?;
                    if arg {
                        return Some(true);
                    }
                }

                let res_var = self.expr_vars[expr_id as usize];
                self.check_ty_never(res_var)
            }

            // the cond may contribute to a never result
            // then check the type of the expression
            Expr::If(cond,_,_) => {
                let cond = self.check_expr_is_never(func, *cond)?;
                if cond {
                    return Some(true);
                }

                let res_var = self.expr_vars[expr_id as usize];
                self.check_ty_never(res_var)
            }
            // check the type of the expression
            Expr::Block(_) | Expr::Loop(_) => {
                let res_var = self.expr_vars[expr_id as usize];
                self.check_ty_never(res_var)
            }

            // never diverge
            Expr::While(..) | // <- looks like the cond can just be ignored
            Expr::DeclVar(_) | Expr::Var(..) | Expr::CastPrimitive(..) | Expr::DeclTmp(_) |
            Expr::LitInt(..) | Expr::LitFloat(..) | Expr::LitBool(_) | Expr::LitChar(_) | Expr::LitVoid => Some(false),

            // always diverge
            Expr::Return(_) => Some(true),

            _ => panic!("todo never check {:?}",expr)
        }
    }

    fn check_ty_never(&self, var: TypeVar) -> Option<bool> {
        // check if this type or any fields or arguments contain unknown or never
        // for these purposes, unknown int and float vars do not count as unknown
        let ty = self.get(var);
        for arg in &ty.args {
            let res = self.check_ty_never(*arg)?;
            if res {
                return Some(true);
            }
        }
        match ty.kind {
            TypeKind::Never => Some(true),
            TypeKind::Unknown => None,
            _ => Some(false)
        }
    }

    fn check_known_strict(&self, var: TypeVar) -> bool {
        let ty = self.get(var);
        if !ty.kind.is_known_strict() {
            return false;
        }
        for arg in &ty.args {
            if !self.check_known_strict(*arg) {
                return false;
            }
        }
        true
    }

    pub fn fix_unknown_primitives(&mut self) {
        for ty in &mut self.var_types {
            if let TypeKind::Int(None) = ty.kind {
                ty.kind = TypeKind::Int(Some((IntWidth::Int32,IntSign::Signed)));
            } else if let TypeKind::Float(None) = ty.kind {
                ty.kind = TypeKind::Float(Some(FloatWidth::Float64));
            }
        }
    }

    pub fn apply_types(&self, func: &mut FuncHIR) {
        assert_eq!(func.exprs.len(),self.expr_vars.len());
        for (expr,var) in func.exprs.iter_mut().zip(self.expr_vars.iter()) {
            if *var == TYPE_INVALID {
                println!("no type for expr {:?}",expr);
                panic!();
            }
            expr.ty = self.get_global(*var);
        }
    }

    pub fn constraint_dump(&self) {
        println!("=== UNSOLVED ===");
        for c in &self.constraints {
            self.dump_constraint(c);
        }
        println!("=== SOLVED ===");
        for c in &self.constraints_sat {
            self.dump_constraint(c);
        }
    }

    fn dump_constraint(&self, c: &TypeConstraint) {
        match c {
            TypeConstraint::Assign{ src, dst } => {
                print!("[assign] ");
                self.dump_var(*src);
                print!(" -> ");
                self.dump_var(*dst);
                println!();
            }
            TypeConstraint::LeastUpperBound2{ res, arg1, arg2 } => {
                print!("[lub2] ");
                self.dump_var(*arg1);
                print!(" + ");
                self.dump_var(*arg2);
                print!(" -> ");
                self.dump_var(*res);
                println!();
            }
            _ => println!("!!! todo dump {:?}",c)
        }
    }

    fn dump_var(&self, var: TypeVar) {
        print!("({}: {:?})",var.0,self.get(var));
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }
}
