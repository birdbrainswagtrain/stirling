
use crate::{hir_expr::{FuncCode, Expr}, types::{Type, TypeInt}};

#[derive(Clone,Copy)]
struct CheckResult{
    mutated: bool,
    resolved: bool
}

impl CheckResult {
    fn combine(self, other: Self) -> Self {
        CheckResult{
            mutated: self.mutated || other.mutated,
            resolved: self.resolved && other.resolved
        }
    }

    fn set_mutated(&mut self) -> Self {
        self.mutated = true;
        *self
    }
}

impl FuncCode {
    pub fn check(&mut self) {

        let mut step = 0;

        loop {
            println!("=================== STEP {} ===================",step);
            step += 1;
            self.print();

            if !self.check_step( 0..self.exprs.len() ) {
                break;
            }

            println!("=================== STEP {} ===================",step);
            step += 1;
            self.print();

            if !self.check_step( (0..self.exprs.len()).rev() ) {
                break;
            }
        }

        for i in 0..self.exprs.len() {
            let ty = self.exprs[i].ty;
            if ty.is_unknown() {
                if ty == Type::IntUnknown {
                    self.exprs[i].ty = Type::Int(TypeInt::I32);
                } else {
                    panic!("function contains unknown type");
                }
            }
        }
        println!("=================== FINAL ===================");
        self.print();
    }

    fn check_step<T: std::iter::Iterator<Item = usize> >(&mut self, iter: T) -> bool
    {
        let mut mutated = false;

        for i in iter {
            if !self.exprs[i].is_resolved {
                let check_res = self.check_expr(i as u32);

                self.exprs[i].is_resolved = check_res.resolved;
                if check_res.mutated {
                    mutated = true;
                }
            }
        }

        mutated
    }

    // returns true if anything was mutated
    fn check_expr(&mut self, index: u32) -> CheckResult {
        let info = &self.exprs[index as usize];

        match info.expr {
            Expr::LitInt(_) | Expr::Var(_) | Expr::LitBool(_) => {
                // no-ops
                CheckResult{mutated: false, resolved: true}
            },
            Expr::Assign(dst,src) => {
                self.check_match_3(index, dst, src)
            },
            Expr::BinOpPrimitive(lhs,op,rhs) => {
                self.check_bin_op(index, lhs,op,rhs)
            },
            Expr::BinOp(lhs,op,rhs) => {
                let lty = self.exprs[lhs as usize].ty;
                let rty = self.exprs[rhs as usize].ty;

                let is_primitive = match op_kind(&op) {
                    OpKind::Arithmetic |
                    OpKind::Ord => lty.is_number() && rty.is_number(),
                    OpKind::Eq => panic!("eq")
                };

                if is_primitive {
                    self.exprs[index as usize].expr = Expr::BinOpPrimitive(lhs,op,rhs);
                    self.check_bin_op(index, lhs, op, rhs).set_mutated()
                } else {
                    panic!("todo more binary stuff");
                }
            },

            Expr::UnOp(arg,op) => {
                let arg_ty = self.exprs[arg as usize].ty;

                let is_primitive = match op {
                    syn::UnOp::Neg(_) => arg_ty.is_number(),
                    syn::UnOp::Not(_) => arg_ty.is_int() || arg_ty == Type::Bool,
                    _ => panic!("unexpected unary operator")
                };

                if is_primitive {
                    self.exprs[index as usize].expr = Expr::UnOpPrimitive(arg,op);
                    self.check_match_2(index, arg).set_mutated()
                } else {
                    panic!("todo more unary stuff");
                }
            },
            Expr::UnOpPrimitive(arg,_op) => {
                // should always be the right play
                self.check_match_2(index, arg)
            },

            Expr::Block(ref block) => {
                if let Some(result_id) = block.result {
                    self.check_match_2(index, result_id)
                } else {
                    panic!("set void");
                }
            },
            Expr::IfElse(cond,ref then_block, else_expr) => {
                let then_expr = then_block.result;

                let mutated = self.update_expr_type(cond,Type::Bool);

                let mut res = if let Some(then_expr) = then_expr {
                    self.check_match_3(index, then_expr, else_expr)
                } else {
                    // else side must be void
                    /*if self.update_expr_type(else_expr as usize, Type::Void) {
                        result = true;
                    }
                    if self.exprs[index].ty.can_upgrade_to(Type::Void) {
                        self.exprs[index].ty = Type::Void;
                    }*/
                    panic!("stop");
                };

                if mutated {
                    res.set_mutated();
                }

                res
            },
            _ => panic!("todo check {:?}",info.expr)
        }
    }

    fn check_bin_op(&mut self, index: u32, lhs: u32, op: syn::BinOp, rhs: u32) -> CheckResult {

        let op_kind = op_kind(&op);

        match op_kind {
            OpKind::Arithmetic => self.check_match_3(index, lhs, rhs),
            OpKind::Ord | OpKind::Eq => {
                let mutated = self.update_expr_type(index, Type::Bool);
                let mut res = self.check_match_2(lhs, rhs);
                if mutated {
                    res.set_mutated();
                }
                res
            }
        }
    }

    fn check_match_2_internal(&mut self, arg1: u32, arg2: u32) -> bool {
        let arg1_ty = self.exprs[arg1 as usize].ty;
        let arg2_ty = self.exprs[arg2 as usize].ty;

        let mutated = if arg1_ty != arg2_ty {
            if arg1_ty.can_upgrade_to(arg2_ty) {
                self.exprs[arg1 as usize].ty = arg2_ty;
            } else {
                self.exprs[arg2 as usize].ty = arg1_ty;
            }
            true
        } else {
            false
        };

        mutated
    }

    fn check_match_2(&mut self, arg1: u32, arg2: u32) -> CheckResult {
        let mutated = self.check_match_2_internal(arg1, arg2);

        let resolved = 
            !self.exprs[arg1 as usize].ty.is_unknown() &&
            !self.exprs[arg2 as usize].ty.is_unknown();

        CheckResult{mutated, resolved}
    }

    fn check_match_3(&mut self, arg1: u32, arg2: u32, arg3: u32) -> CheckResult {
        // check each pair
        let mutated_1 = self.check_match_2_internal(arg1,arg2);
        let mutated_2 = self.check_match_2_internal(arg2,arg3);
        let mutated_3 = self.check_match_2_internal(arg1,arg3);

        let mutated = mutated_1 || mutated_2 || mutated_3;

        let resolved = 
            !self.exprs[arg1 as usize].ty.is_unknown() &&
            !self.exprs[arg2 as usize].ty.is_unknown() &&
            !self.exprs[arg3 as usize].ty.is_unknown();

        CheckResult{mutated, resolved}
    }

    // returns mutated
    fn update_expr_type(&mut self, index: u32, ty: Type) -> bool {
        let old_ty = self.exprs[index as usize].ty;
        if ty != old_ty {
            if old_ty.can_upgrade_to(ty) {
                self.exprs[index as usize].ty = ty;
            } else {
                panic!("can not upgrade type");
            }
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
enum OpKind {
    Arithmetic,
    // apply to floats and ints
    // argument types and result type all match

    Ord,
    // apply to floats and ints
    // argument types must match but result type must be bool

    Eq,
    // apply to any primitives
    // argument types must match but result type must be bool
}

fn op_kind(op: &syn::BinOp) -> OpKind {
    use syn::BinOp;
    match op {
        BinOp::Add(_) | BinOp::AddEq(_) |
        BinOp::Sub(_) | BinOp::SubEq(_) |
        BinOp::Mul(_) | BinOp::MulEq(_) |
        BinOp::Div(_) | BinOp::DivEq(_) |
        BinOp::Rem(_) | BinOp::RemEq(_) => OpKind::Arithmetic,

        BinOp::Lt(_) | BinOp::Gt(_) |
        BinOp::Le(_) | BinOp::Ge(_) => OpKind::Ord,

        _ => panic!("todo op kind {:?}",op)
    }
}


    /*fn update_expr_type(&mut self, index: usize, ty: Type) -> bool {
        if self.exprs[index].ty.can_upgrade_to(ty) {
            self.exprs[index].ty = ty;
            
            let info = &self.exprs[index];
            match info.expr {
                Expr::Var(_x) => (),
                Expr::Block(ref block) => {
                    if let Some(res) = block.result {
                        self.update_expr_type(res as usize, ty);
                    } else {
                        panic!("no result, asset ty = void");
                    }
                },
                Expr::IfElse(_cond,ref then_block, else_expr) => {
                    if let Some(then_expr) = then_block.result {
                        self.update_expr_type(then_expr as usize, ty);
                    } else {
                        panic!("no result, asset ty = void");
                    }
                    self.update_expr_type(else_expr as usize, ty);
                },
                Expr::LitInt(_x) => {
                    assert!(ty.is_int());
                },
                Expr::BinOp(..) => (), // can't do anything at this stage
                Expr::BinOpPrimitive(lhs,_op,rhs) => {
                    // todo this is NOT correct for all operators
                    self.update_expr_type(lhs as usize, ty);
                    self.update_expr_type(rhs as usize, ty);
                },
                Expr::UnOp(..) => (),
                Expr::UnOpPrimitive(arg,_op) => {
                    self.update_expr_type(arg as usize, ty);
                },
                _ => panic!("todo update {:?}",info.expr)
            }
            return true;
        }
        false
    }*/
