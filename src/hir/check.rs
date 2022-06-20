use crate::builtin::BUILTINS;
use crate::is_verbose;

use super::func::{Expr, FuncHIR, Block};
use super::types::{CompoundType, Signature, Type};

#[derive(Clone, Copy)]
struct CheckResult {
    mutated: bool,
    resolved: bool,
}

impl CheckResult {
    fn combine(self, other: Self) -> Self {
        CheckResult {
            mutated: self.mutated || other.mutated,
            resolved: self.resolved && other.resolved,
        }
    }

    fn set_mutated(&mut self) -> Self {
        self.mutated = true;
        *self
    }
}

impl FuncHIR {
    pub fn check(&mut self) {
        let mut step = 0;

        loop {
            if is_verbose() {
                println!("=================== STEP {} ===================", step);
                step += 1;
                self.print();
            }

            if self.check_step(0..self.exprs.len()) {
                break;
            }

            if is_verbose() {
                println!("=================== STEP {} ===================", step);
                step += 1;
                self.print();
            }

            if self.check_step((0..self.exprs.len()).rev()) {
                break;
            }
        }

        for i in 0..self.exprs.len() {
            self.exprs[i].ty.check_known();
        }
        if is_verbose() {
            println!("=================== FINAL ===================");
            self.print();
        }
    }

    fn check_step<T: std::iter::Iterator<Item = usize>>(&mut self, iter: T) -> bool {
        let mut mutated = false;
        let mut resolved = true;

        for i in iter {
            if !self.exprs[i].is_resolved {
                let check_res = self.check_expr(i as u32);

                self.exprs[i].is_resolved = check_res.resolved;
                mutated = mutated || check_res.mutated;
                resolved = resolved && check_res.resolved;
            }
        }

        resolved || !mutated
    }

    // returns true if anything was mutated
    fn check_expr(&mut self, index: u32) -> CheckResult {
        let info = &self.exprs[index as usize];

        match info.expr {
            Expr::Var(..)
            | Expr::LitInt(_)
            | Expr::LitFloat(_)
            | Expr::LitChar(_)
            | Expr::LitBool(_)
            | Expr::LitVoid
            | Expr::DeclVar(_)
            | Expr::DeclTmp(_)
            | Expr::StmtTmp(_, _)
            | Expr::CastPrimitive(_)
            => {
                // no-ops
                CheckResult {
                    mutated: false,
                    resolved: true,
                }
            }
            Expr::Ref(arg, is_mut) => {
                let arg_ty = self.exprs[arg as usize].ty;

                let ref_ty = if let Type::Compound(CompoundType::Ref(ref_ty, ref_mut)) =
                    self.exprs[index as usize].ty
                {
                    assert_eq!(is_mut, *ref_mut);
                    *ref_ty
                } else {
                    Type::Unknown
                };

                if ref_ty == arg_ty {
                    CheckResult {
                        mutated: false,
                        resolved: !ref_ty.is_unknown(),
                    }
                } else {
                    let new_ty = ref_ty.unify(arg_ty);

                    self.exprs[arg as usize].ty = new_ty;
                    self.exprs[index as usize].ty =
                        Type::from_compound(CompoundType::Ref(new_ty, is_mut));

                    CheckResult {
                        mutated: true,
                        resolved: !new_ty.is_unknown(),
                    }
                }
            }
            Expr::DeRef(arg) => {
                let arg_ty = self.exprs[arg as usize].ty;
                let mut mutated = false;
                // deref can be overridden, so we can't just assume the arg type is a ref
                if let Type::Compound(CompoundType::Ref(ref_ty, ref_mut)) = arg_ty {
                    if info.ty != *ref_ty {
                        let new_ty = info.ty.unify(*ref_ty);

                        self.exprs[index as usize].ty = new_ty;
                        self.exprs[arg as usize].ty =
                            Type::from_compound(CompoundType::Ref(new_ty, *ref_mut));

                        mutated = true;
                    }
                }
                let r1 = !self.exprs[index as usize].ty.is_unknown();
                let r2 = !self.exprs[arg as usize].ty.is_unknown();

                CheckResult {
                    mutated,
                    resolved: r1 && r2,
                }
            }
            Expr::Assign(dst, src) => {
                self.check_match_2(dst, src)
            }
            Expr::BinOpPrimitive(lhs, op, rhs) => self.check_bin_op(index, lhs, op, rhs),
            Expr::BinOp(lhs, op, rhs) => {
                let lty = self.exprs[lhs as usize].ty;
                let rty = self.exprs[rhs as usize].ty;

                let is_primitive = match op_class(&op) {
                    OpClass::Arithmetic => lty.is_number() && rty.is_number(),
                    OpClass::Ord => {
                        (lty.is_number() && rty.is_number())
                            || (lty == Type::Char && rty == Type::Char)
                    }
                    OpClass::Bitwise => {
                        (lty.is_int() && rty.is_int()) || (lty == Type::Bool && rty == Type::Bool)
                    }
                    OpClass::Logical => true,
                    OpClass::Eq => lty.is_prim_eq() && rty.is_prim_eq(),
                    OpClass::BitShift => lty.is_int() && rty.is_int(),
                };

                if is_primitive {
                    self.exprs[index as usize].expr = Expr::BinOpPrimitive(lhs, op, rhs);
                    self.check_bin_op(index, lhs, op, rhs).set_mutated()
                } else {
                    //panic!("todo more binary stuff {:?} {:?} {:?}", lty, op, rty);
                    CheckResult {
                        mutated: false,
                        resolved: false,
                    }
                }
            }

            Expr::UnOp(arg, op) => {
                let arg_ty = self.exprs[arg as usize].ty;

                let is_primitive = match op {
                    syn::UnOp::Neg(_) => arg_ty.is_number(),
                    syn::UnOp::Not(_) => arg_ty.is_int() || arg_ty == Type::Bool,
                    _ => panic!("unexpected unary operator"),
                };

                if is_primitive {
                    self.exprs[index as usize].expr = Expr::UnOpPrimitive(arg, op);
                    self.check_match_2(index, arg).set_mutated()
                } else {
                    panic!("todo more unary stuff");
                }
            }
            Expr::UnOpPrimitive(arg, _op) => {
                // should always be the right play
                self.check_match_2(index, arg)
            }

            Expr::Block(ref block) => {
                if let Some(result_id) = block.result {
                    self.check_match_2(index, result_id)
                } else {
                    if let Some(is_never) = self.get_block_is_never(block) {
                        let ty = if is_never { Type::Never } else { Type::Void };
                        let mutated = self.update_expr_type(index, ty);
                        CheckResult {
                            mutated,
                            resolved: true
                        }
                    } else {
                        CheckResult {
                            mutated: false,
                            resolved: false
                        }
                    }
                }
            }
            Expr::If(cond, ref then_block, else_expr) => {
                let then_expr = then_block.result;

                let mut res = if let Some(else_expr) = else_expr {
                    // if-then-else
                    if let Some(then_expr) = then_expr {
                        self.check_match_3(index, then_expr, else_expr)
                    } else {
                        // else side must be void or never
                        let ty_then = match self.get_block_is_never(then_block) {
                            Some(true) => Type::Never,
                            Some(false) => Type::Void,
                            None => Type::Unknown
                        };

                        let mutated = self.update_expr_type(index, ty_then);
                        let mut result = self.check_match_2(index, else_expr);
                        if mutated {
                            result.set_mutated();
                        }
                        result
                    }
                } else {
                    // if-then
                    let m1 = if let Some(then_expr) = then_expr {
                        self.update_expr_type(then_expr, Type::Void)
                    } else {
                        false
                    };

                    let m2 = self.update_expr_type(index, Type::Void);

                    CheckResult {
                        mutated: m1 || m2,
                        resolved: true,
                    }
                };

                let mutated = self.update_expr_type(cond, Type::Bool);
                if mutated {
                    res.set_mutated();
                }

                res
            }
            Expr::While(cond, ref body_block) => {
                let body_expr = body_block.result;

                let m1 = self.update_expr_type(cond, Type::Bool);
                let m2 = if let Some(body_expr) = body_expr {
                    self.update_expr_type(body_expr, Type::Void)
                } else {
                    false
                };

                CheckResult {
                    mutated: m1 || m2,
                    resolved: true,
                }
            }
            Expr::Loop(ref body_block) => {
                let body_expr = body_block.result;

                let mutated = if let Some(body_expr) = body_expr {
                    self.update_expr_type(body_expr, Type::Void)
                } else {
                    false
                };

                CheckResult {
                    mutated,
                    resolved: true,
                }
            }
            Expr::Break(target_loop, value) => {
                let loop_expr = &self.exprs[target_loop as usize].expr;

                if let Expr::Loop(_) = loop_expr {
                    // loop loops require type checking
                    if let Some(value) = value {
                        self.check_match_2(target_loop, value)
                    } else {
                        let mutated = self.update_expr_type(target_loop, Type::Void);
                        CheckResult {
                            mutated,
                            resolved: true,
                        }
                    }
                } else {
                    assert!(value.is_none());
                    CheckResult {
                        mutated: false,
                        resolved: true,
                    }
                }
            }
            Expr::Continue(_target_loop) => CheckResult {
                mutated: false,
                resolved: true,
            },
            Expr::Return(arg) => {
                if let Some(arg) = arg {
                    // TODO lambdas might not have a known return type :(
                    self.check_match_2(arg, self.root_expr as u32)
                } else {
                    CheckResult {
                        mutated: false,
                        resolved: true,
                    }
                }
            }
            Expr::CallBuiltin(ref name, ref args) => {
                let entry = BUILTINS.get(name.as_str());
                if entry.is_none() {
                    panic!("invalid builtin: {}", name);
                }

                let sig = &entry.unwrap().1;
                let args = args.clone(); // TODO BAD CLONE

                self.check_function(index, &args, sig)
            }
            Expr::Call(func, ref args) => {
                let sig = func.sig();
                let args = args.clone(); // TODO BAD CLONE

                self.check_function(index, &args, sig)
            }
            _ => panic!("todo check {:?}", info.expr),
        }
    }

    fn check_function(&mut self, index: u32, args: &Vec<u32>, sig: &Signature) -> CheckResult {
        assert_eq!(args.len(), sig.inputs.len());

        let mut mutated = false;

        for (arg, ty) in args.iter().zip(&sig.inputs) {
            if self.update_expr_type(*arg, *ty) {
                mutated = true;
            }
        }

        if self.update_expr_type(index, sig.output) {
            mutated = true;
        }

        CheckResult {
            mutated,
            resolved: true,
        }
    }

    fn get_expr_is_never(&self, index: u32) -> Option<bool> {

        let info = &self.exprs[index as usize];
        match &info.expr {
            // never never
            Expr::DeclVar(_) | Expr::Var(_) | Expr::DeclTmp(_) |
            Expr::LitBool(_) | Expr::LitInt(_) | Expr::LitFloat(_) | Expr::LitVoid | Expr::LitChar(_) => Some(false),


            // always never
            Expr::Return(_) => Some(true),

            // sometimes never
            Expr::Block(block) => {
                self.get_block_is_never(block)
            }
            Expr::If(arg, then_block, else_expr) => {
                if self.get_expr_is_never(*arg)? {
                    return Some(true);
                }

                // if is only never if both branches are
                if let Some(else_expr) = else_expr {
                    let a = self.get_block_is_never(then_block)?;
                    let b = self.get_expr_is_never(*else_expr)?;

                    Some(a && b)
                } else {
                    Some(false)
                }
            }
            Expr::While(arg, _) => {
                self.get_expr_is_never(*arg)
            }
            Expr::Loop(_) => {
                // currently we scan the entire expression list for breaks
                for expr in &self.exprs {
                    if let Expr::Break(target,_) = expr.expr {
                        if target == index {
                            return Some(false);
                        }
                    }
                }
                Some(true)
            }
            Expr::Call(func, args) => {
                for arg in args {
                    if self.get_expr_is_never(*arg)? {
                        return Some(true);
                    }
                }

                let sig = func.sig();
                Some(sig.output == Type::Never)
            }
            Expr::CallBuiltin(name, args) => {
                for arg in args {
                    if self.get_expr_is_never(*arg)? {
                        return Some(true);
                    }
                }
                
                let entry = BUILTINS.get(name.as_str());
                if entry.is_none() {
                    panic!("invalid builtin: {}", name);
                }

                let sig = &entry.unwrap().1;
                Some(sig.output == Type::Never)
            }

            Expr::StmtTmp(arg, _) |
            Expr::UnOp(arg, _) |
            Expr::UnOpPrimitive(arg, _) |
            Expr::Ref(arg, _) |
            Expr::DeRef(arg) |
            Expr::CastPrimitive(arg) => {
                self.get_expr_is_never(*arg)
            }

            Expr::Assign(a, b) |
            Expr::BinOp(a, _, b) |
            Expr::BinOpPrimitive(a, _, b) => {
                if self.get_expr_is_never(*a)? {
                    return Some(true);
                }

                self.get_expr_is_never(*b)
            }
            
            _ => panic!("todo never-check {:?}",info.expr)
        }
    }

    fn get_block_is_never(&self, block: &Block) -> Option<bool> {

        if let Some(res) = block.result {
            if self.get_expr_is_never(res)? {
                return Some(true);
            }
        }

        for stmt_id in &block.stmts {
            if self.get_expr_is_never(*stmt_id)? {
                return Some(true);
            }
        }

        Some(false)
    }

    fn check_bin_op(&mut self, index: u32, lhs: u32, op: syn::BinOp, rhs: u32) -> CheckResult {
        let is_assign = op_is_assign(&op);
        let mut res = match op_class(&op) {
            OpClass::Arithmetic | OpClass::Bitwise => {
                if is_assign {
                    self.check_match_2(lhs, rhs)
                } else {
                    self.check_match_3(index, lhs, rhs)
                }
            }
            OpClass::BitShift => {
                if is_assign {
                    CheckResult {
                        mutated: false,
                        resolved: true,
                    }
                } else {
                    // only lhs must match
                    self.check_match_2(index, lhs)
                }
            }
            OpClass::Ord | OpClass::Eq => {
                assert!(!is_assign);
                let mutated = self.update_expr_type(index, Type::Bool);
                let mut res = self.check_match_2(lhs, rhs);
                if mutated {
                    res.set_mutated();
                }
                res
            }
            OpClass::Logical => {
                assert!(!is_assign);
                let m1 = self.update_expr_type(index, Type::Bool);
                let m2 = self.update_expr_type(lhs, Type::Bool);
                let m3 = self.update_expr_type(rhs, Type::Bool);
                CheckResult {
                    mutated: m1 || m2 || m3,
                    resolved: true,
                }
            }
        };

        if is_assign {
            if self.update_expr_type(index, Type::Void) {
                res.set_mutated();
            }
        }

        res
    }

    fn check_match_2_internal(&mut self, arg1: u32, arg2: u32) -> bool {
        let arg1_ty = self.exprs[arg1 as usize].ty;
        let arg2_ty = self.exprs[arg2 as usize].ty;

        let mutated = if arg1_ty != arg2_ty {
            let new_ty = arg1_ty.unify(arg2_ty);
            self.exprs[arg1 as usize].ty = new_ty;
            self.exprs[arg2 as usize].ty = new_ty;
            true
        } else {
            false
        };

        mutated
    }

    fn check_match_2(&mut self, arg1: u32, arg2: u32) -> CheckResult {
        let mutated = self.check_match_2_internal(arg1, arg2);

        let resolved = !self.exprs[arg1 as usize].ty.is_unknown()
            && !self.exprs[arg2 as usize].ty.is_unknown();

        CheckResult { mutated, resolved }
    }

    fn check_match_3(&mut self, arg1: u32, arg2: u32, arg3: u32) -> CheckResult {
        // check each pair
        let mutated_1 = self.check_match_2_internal(arg1, arg2);
        let mutated_2 = self.check_match_2_internal(arg2, arg3);
        let mutated_3 = self.check_match_2_internal(arg1, arg3);

        let mutated = mutated_1 || mutated_2 || mutated_3;

        let resolved = !self.exprs[arg1 as usize].ty.is_unknown()
            && !self.exprs[arg2 as usize].ty.is_unknown()
            && !self.exprs[arg3 as usize].ty.is_unknown();

        CheckResult { mutated, resolved }
    }

    // returns mutated
    fn update_expr_type(&mut self, index: u32, ty: Type) -> bool {
        let old_ty = self.exprs[index as usize].ty;
        if ty != old_ty {
            self.exprs[index as usize].ty = old_ty.unify(ty);
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
enum OpClass {
    Arithmetic,
    // apply to floats and ints
    // argument types and result type all match
    Ord,
    // apply to floats and ints, do NOT apply to bools even though they are ord
    // argument types must match but result type must be bool
    Eq,
    // apply to all primitives
    // argument types must match but result type must be bool
    Bitwise,
    // apply to ints and bools
    // argument types and result type all match
    Logical,
    // apply to bools
    // all types must be bools
    BitShift,
    // TODO
}

fn op_class(op: &syn::BinOp) -> OpClass {
    use syn::BinOp;
    match op {
        BinOp::Add(_)
        | BinOp::AddEq(_)
        | BinOp::Sub(_)
        | BinOp::SubEq(_)
        | BinOp::Mul(_)
        | BinOp::MulEq(_)
        | BinOp::Div(_)
        | BinOp::DivEq(_)
        | BinOp::Rem(_)
        | BinOp::RemEq(_) => OpClass::Arithmetic,

        BinOp::Lt(_) | BinOp::Gt(_) | BinOp::Le(_) | BinOp::Ge(_) => OpClass::Ord,

        BinOp::Eq(_) | BinOp::Ne(_) => OpClass::Eq,

        BinOp::BitAnd(_)
        | BinOp::BitAndEq(_)
        | BinOp::BitOr(_)
        | BinOp::BitOrEq(_)
        | BinOp::BitXor(_)
        | BinOp::BitXorEq(_) => OpClass::Bitwise,

        BinOp::Shl(_) | BinOp::ShlEq(_) | BinOp::Shr(_) | BinOp::ShrEq(_) => OpClass::BitShift,

        BinOp::Or(_) | BinOp::And(_) => OpClass::Logical,
    }
}

fn op_is_assign(op: &syn::BinOp) -> bool {
    use syn::BinOp;
    match op {
        BinOp::AddEq(_)
        | BinOp::SubEq(_)
        | BinOp::MulEq(_)
        | BinOp::DivEq(_)
        | BinOp::RemEq(_)
        | BinOp::BitAndEq(_)
        | BinOp::BitOrEq(_)
        | BinOp::BitXorEq(_)
        | BinOp::ShlEq(_)
        | BinOp::ShrEq(_) => true,

        BinOp::Add(_)
        | BinOp::Sub(_)
        | BinOp::Mul(_)
        | BinOp::Div(_)
        | BinOp::Rem(_)
        | BinOp::BitAnd(_)
        | BinOp::BitOr(_)
        | BinOp::BitXor(_)
        | BinOp::Shl(_)
        | BinOp::Shr(_)
        | BinOp::Lt(_)
        | BinOp::Gt(_)
        | BinOp::Le(_)
        | BinOp::Ge(_)
        | BinOp::Eq(_)
        | BinOp::Ne(_)
        | BinOp::Or(_)
        | BinOp::And(_) => false,
    }
}
