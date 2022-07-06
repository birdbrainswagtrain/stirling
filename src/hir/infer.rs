use std::rc::Rc;

use syn::{BinOp, UnOp};

use crate::builtin::BUILTINS;

use super::{func::{Block, Expr, FuncHIR, ExprInfo}, types::{Type, IntType}};

#[derive(Debug,Clone,Copy,PartialEq)]
pub struct TypeVar(u32);

#[derive(Debug,Clone,Copy,PartialEq)]
enum IntWidth {
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    IntSize
}

#[derive(Debug,Clone,Copy,PartialEq)]
enum IntSign {
    Signed,
    Unsigned
}

fn convert_legacy_ty(ty: &Type) -> TypeKind {
    match ty {

        Type::Int(IntType::I32) => TypeKind::Int(Some((IntWidth::Int32,IntSign::Signed))),
        Type::Int(IntType::U32) => TypeKind::Int(Some((IntWidth::Int32,IntSign::Unsigned))),

        Type::Int(IntType::I128) => TypeKind::Int(Some((IntWidth::Int128,IntSign::Signed))),
        Type::Int(IntType::U128) => TypeKind::Int(Some((IntWidth::Int128,IntSign::Unsigned))),

        Type::Bool => TypeKind::Bool,
        Type::Unknown => TypeKind::Unknown,
        Type::Void => TypeKind::Tuple,
        _ => panic!("todo convert {:?}",ty)
    }
}

fn global_type_from_legacy(ty: &Type) -> GlobalType {
    let kind = convert_legacy_ty(ty);
    GlobalType{ kind, args: None }
}

#[derive(Debug,Clone,PartialEq)]
pub enum TypeKind {
    Unknown,
    Never,
    Tuple,
    Int(Option<(IntWidth,IntSign)>),
    Bool
}

impl TypeKind {
    fn is_known(&self) -> bool {
        match self {
            TypeKind::Unknown | TypeKind::Int(None) => false,
            _ => true
        }
    }
}

#[derive(Debug,Clone,PartialEq)]
struct LocalType {
    kind: TypeKind,
    args: Vec<TypeVar>
}

impl LocalType {
    fn set_unify(&mut self, other: &LocalType) {
        assert_eq!(self.args.len(), 0);
        assert_eq!(other.args.len(), 0);

        match (&self.kind,&other.kind) {
            // our type is more specific
            (_,TypeKind::Unknown) => (),
            (TypeKind::Int(Some(_)),TypeKind::Int(None)) => (),

            _ => panic!("todo unify kinds {:?} {:?}",self.kind,other.kind)
        }
    }
}

//type GlobalType = Rc<GlobalTypeData>;

#[derive(Debug)]
pub struct GlobalType {
    kind: TypeKind,
    args: Option<Rc<[GlobalType]>>
}

impl GlobalType {
    pub fn simple(kind: TypeKind) -> Self {
        GlobalType{kind, args: None}
    }
}

#[derive(Debug)]
enum TypeConstraint {
    Equal(TypeVar,TypeVar),
    EqualLit(TypeVar,GlobalType),
    Op{op: OpConstraint, res: TypeVar, lhs: TypeVar, rhs: TypeVar},
    UnOp{op: UnOpConstraint, res: TypeVar, arg: TypeVar}
}

#[derive(Debug,Clone,Copy)]
enum UnOpConstraint {
    Neg,
    Not
}

#[derive(Debug,Clone,Copy)]
enum OpConstraint {
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

fn op_constraint_and_type(op: &BinOp) -> (OpConstraint,TypeKind) {
    match op {
        BinOp::Lt(_) | BinOp::Gt(_) | BinOp::Le(_) | BinOp::Ge(_) => (OpConstraint::Ord,TypeKind::Bool),

        BinOp::Eq(_) | BinOp::Ne(_) => (OpConstraint::Eq,TypeKind::Bool),

        BinOp::Add(_) => (OpConstraint::Add,TypeKind::Unknown),
        BinOp::Sub(_) => (OpConstraint::Sub,TypeKind::Unknown),
        BinOp::Mul(_) => (OpConstraint::Mul,TypeKind::Unknown),
        BinOp::Div(_) => (OpConstraint::Div,TypeKind::Unknown),
        BinOp::Rem(_) => (OpConstraint::Rem,TypeKind::Unknown),

        BinOp::BitOr(_) => (OpConstraint::BitOr,TypeKind::Unknown),
        BinOp::BitAnd(_) => (OpConstraint::BitAnd,TypeKind::Unknown),
        BinOp::BitXor(_) => (OpConstraint::BitXor,TypeKind::Unknown),

        BinOp::Shl(_) => (OpConstraint::ShiftL,TypeKind::Unknown),
        BinOp::Shr(_) => (OpConstraint::ShiftR,TypeKind::Unknown),

        BinOp::AddEq(_) => (OpConstraint::AddEq,TypeKind::Unknown),
        BinOp::SubEq(_) => (OpConstraint::SubEq,TypeKind::Unknown),
        BinOp::MulEq(_) => (OpConstraint::MulEq,TypeKind::Unknown),
        BinOp::DivEq(_) => (OpConstraint::DivEq,TypeKind::Unknown),
        BinOp::RemEq(_) => (OpConstraint::RemEq,TypeKind::Unknown),

        BinOp::BitOrEq(_) => (OpConstraint::BitOrEq,TypeKind::Unknown),
        BinOp::BitAndEq(_) => (OpConstraint::BitAndEq,TypeKind::Unknown),
        BinOp::BitXorEq(_) => (OpConstraint::BitXorEq,TypeKind::Unknown),

        BinOp::ShlEq(_) => (OpConstraint::ShiftLEq,TypeKind::Unknown),
        BinOp::ShrEq(_) => (OpConstraint::ShiftREq,TypeKind::Unknown),

        _ => panic!("todo op {:?}",op)
    }
}

pub struct FuncTypes {
    var_types: Vec<LocalType>,
    expr_vars: Vec<TypeVar>,
    constraints: Vec<TypeConstraint>,
    constraints_sat: Vec<TypeConstraint>
}

const TYPE_INVALID: TypeVar = TypeVar(0xFFFFFFFF);
const TYPE_VOID: TypeVar = TypeVar(0);

impl FuncTypes {
    pub fn new(func: &mut FuncHIR) -> Self {
        let mut types = FuncTypes{
            var_types: vec!(
                LocalType{kind: TypeKind::Tuple, args: vec!()}, // TYPE_VOID
            ),
            expr_vars: vec!(TYPE_INVALID; func.exprs.len()),
            constraints: vec!(),
            constraints_sat: vec!()
        };

        types.init_expr(func, func.root_expr as u32);

        types
    }

    pub fn init_expr(&mut self, func: &FuncHIR, expr_id: u32) -> TypeVar {
        let expr = &func.exprs[expr_id as usize].expr;
        let result = match expr {
            Expr::DeclVar(_) => {
                // no-op
                TYPE_VOID
            }
            Expr::Assign(dst, src) => {
                let lhs = self.init_expr(func,*dst);
                let rhs = self.init_expr(func,*src);
                
                self.add_constraint(TypeConstraint::Equal(lhs, rhs));

                TYPE_VOID
            }
            Expr::BinOp(lhs, op, rhs) => {
                let lhs = self.init_expr(func,*lhs);
                let rhs = self.init_expr(func,*rhs);

                let (op,ty) = op_constraint_and_type(op);
                let res = self.new_var(ty);

                self.add_constraint(TypeConstraint::Op{op, res, lhs, rhs});

                res
            }
            Expr::UnOp(arg, op) => {
                let arg = self.init_expr(func,*arg);
                let res = self.new_var(TypeKind::Unknown);
                match op {
                    UnOp::Not(_) => self.add_constraint(TypeConstraint::UnOp{op:UnOpConstraint::Not, res,arg}),
                    UnOp::Neg(_) => self.add_constraint(TypeConstraint::UnOp{op:UnOpConstraint::Neg, res,arg}),
                    UnOp::Deref(_) => panic!()
                }
                res
            }
            Expr::Var(_,ty) => {
                let kind = convert_legacy_ty(ty);
                self.new_var(kind)
            }
            Expr::CastPrimitive(arg, ty) => {
                self.init_expr(func,*arg);

                let kind = convert_legacy_ty(ty);
                self.new_var(kind)
            }
            Expr::LitInt(_,ty) => {
                if *ty != Type::IntUnknown {
                    panic!("todo");
                }
                self.new_var(TypeKind::Int(None))
            }
            Expr::Block(block) => {
                let block_var = self.new_var(TypeKind::Unknown);

                for stmt in &block.stmts {
                    self.init_expr(func,*stmt);
                }

                if let Some(res) = block.result {
                    let res_var = self.init_expr(func,res);
                    self.add_constraint(TypeConstraint::Equal(block_var, res_var));
                } else {
                    println!("todo never check");
                }

                block_var
            }
            Expr::While(cond, block) => {
                let cond_var = self.init_expr(func,*cond);
                self.add_constraint(TypeConstraint::EqualLit(cond_var,GlobalType::simple(TypeKind::Bool)));

                for stmt in &block.stmts {
                    self.init_expr(func,*stmt);
                }

                if let Some(res) = block.result {
                    let res_var = self.init_expr(func,res);
                    self.add_constraint(TypeConstraint::Equal(res_var,TYPE_VOID));
                }

                TYPE_VOID
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
                    self.add_constraint(TypeConstraint::EqualLit(var,global_type_from_legacy(ty)));
                }

                self.new_var(convert_legacy_ty(&sig.output))
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

    fn add_constraint(&mut self, c: TypeConstraint) {
        self.constraints.push(c);
    }

    pub fn solve(&mut self) {
        if self.constraints.len() > 0 {
            let mut i = 0;

            while i < self.constraints.len() {

                let mut sat = false;

                let c = &self.constraints[i];

                match c {
                    TypeConstraint::Equal(var1, var2) => {
                        sat = self.solve_equal(*var1,*var2);
                    }
                    TypeConstraint::UnOp{ op, res, arg } => {
                        sat = self.solve_unary(*op,*res,*arg);
                    }
                    _ => panic!("todo solve {:?}",c)
                }

                if sat {
                    if i == self.constraints.len() - 1 {
                        self.constraints.pop().unwrap();
                    } else {
                        self.constraints[i] = self.constraints.pop().unwrap();
                    }
                } else {
                    i += 1;
                }
            }
        }
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

    fn solve_equal(&mut self, var1: TypeVar, var2: TypeVar) -> bool {
        let (t1,t2) = self.get_mut_2(var1, var2);

        if t1 != t2 {
            t1.set_unify(t2);
            *t2 = t1.clone();
        }

        t1.kind.is_known()
    }

    fn solve_unary(&mut self, op: UnOpConstraint, res: TypeVar, arg: TypeVar) -> bool {
        let (arg,res) = self.get_mut_2(arg, res);

        match arg.kind {
            TypeKind::Int(_) | TypeKind::Bool => {
                if arg != res {
                    arg.set_unify(res);
                    *res = arg.clone();
                }
                return arg.kind.is_known();
            },
            _ => () // TODO TRAIT?
        }

        false
    }
}
