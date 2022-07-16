use std::rc::Rc;

use syn::{BinOp, UnOp};

use crate::builtin::BUILTINS;

use super::{func::{Block, Expr, FuncHIR, ExprInfo}, types::{Type, IntType, FloatType}};

#[derive(Debug,Clone,Copy,PartialEq)]
pub struct TypeVar(u32);

#[derive(Debug,Clone,Copy,PartialEq)]
pub enum IntWidth {
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    IntSize
}

#[derive(Debug,Clone,Copy,PartialEq)]
pub enum FloatWidth {
    Float32,
    Float64
}

#[derive(Debug,Clone,Copy,PartialEq)]
pub enum IntSign {
    Signed,
    Unsigned
}

fn convert_legacy_ty(ty: &Type) -> TypeKind {
    match ty {

        Type::Int(IntType::I8) => TypeKind::Int(Some((IntWidth::Int8,IntSign::Signed))),
        Type::Int(IntType::U8) => TypeKind::Int(Some((IntWidth::Int8,IntSign::Unsigned))),

        Type::Int(IntType::I16) => TypeKind::Int(Some((IntWidth::Int16,IntSign::Signed))),
        Type::Int(IntType::U16) => TypeKind::Int(Some((IntWidth::Int16,IntSign::Unsigned))),

        Type::Int(IntType::I32) => TypeKind::Int(Some((IntWidth::Int32,IntSign::Signed))),
        Type::Int(IntType::U32) => TypeKind::Int(Some((IntWidth::Int32,IntSign::Unsigned))),

        Type::Int(IntType::I64) => TypeKind::Int(Some((IntWidth::Int64,IntSign::Signed))),
        Type::Int(IntType::U64) => TypeKind::Int(Some((IntWidth::Int64,IntSign::Unsigned))),

        Type::Int(IntType::I128) => TypeKind::Int(Some((IntWidth::Int128,IntSign::Signed))),
        Type::Int(IntType::U128) => TypeKind::Int(Some((IntWidth::Int128,IntSign::Unsigned))),

        Type::Int(IntType::ISize) => TypeKind::Int(Some((IntWidth::IntSize,IntSign::Signed))),
        Type::Int(IntType::USize) => TypeKind::Int(Some((IntWidth::IntSize,IntSign::Unsigned))),

        Type::Float(FloatType::F32) => TypeKind::Float(Some(FloatWidth::Float32)),
        Type::Float(FloatType::F64) => TypeKind::Float(Some(FloatWidth::Float64)),

        Type::Bool => TypeKind::Bool,
        Type::Char => TypeKind::Char,

        Type::Unknown => TypeKind::Unknown,
        Type::Void => TypeKind::Tuple,
        _ => panic!("todo convert {:?}",ty)
    }
}

#[derive(Debug,Clone,PartialEq)]
pub enum TypeKind {
    Unknown,
    Never,
    Tuple, // used for '()' / void
    Int(Option<(IntWidth,IntSign)>),
    Float(Option<FloatWidth>),
    Bool,
    Char
}

impl TypeKind {
    pub fn ptr() -> TypeKind {
        Self::Int(Some((IntWidth::IntSize,IntSign::Unsigned)))
    }

    fn is_known(&self) -> bool {
        match self {
            TypeKind::Unknown | TypeKind::Int(None) | TypeKind::Float(None) => false,
            _ => true
        }
    }

    fn cannot_coerce(&self) -> bool {
        match self {
            // this should maybe be a blacklist, but use a whitelist for now
            TypeKind::Int(_) => true,
            _ => false
        }
    }

    /// Is the type a primitive for the purposes of operator inference?
    fn is_op_prim(&self) -> bool {
        match self {
            TypeKind::Bool | TypeKind::Int(_) | TypeKind::Float(_) => true,
            _ => false
        }
    }
}

#[derive(Debug,Clone,PartialEq)]
struct LocalType {
    kind: TypeKind,
    args: Vec<TypeVar>
}

enum UnifyResult {
    A,
    B
}

impl LocalType {
    fn unify(&mut self, other: &mut LocalType) {
        assert_eq!(self.args.len(), 0);
        assert_eq!(other.args.len(), 0);

        let res = Self::unify_kind(&self.kind,&other.kind);

        match res {
            UnifyResult::A => other.kind = self.kind.clone(),
            UnifyResult::B => self.kind = other.kind.clone()
        }
    }

    fn unify_g(&mut self, other: &GlobalType) {
        assert_eq!(self.args.len(), 0);
        assert!(other.args.is_none());

        if self.kind == other.kind {
            return;
        }

        let res = Self::unify_kind(&self.kind,&other.kind);

        match res {
            UnifyResult::A => (),
            UnifyResult::B => self.kind = other.kind.clone()
        }
    }

    fn unify_kind(a: &TypeKind, b: &TypeKind) -> UnifyResult {
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
}

#[derive(Debug,Clone)]
pub struct GlobalType {
    pub kind: TypeKind,
    pub args: Option<Rc<[GlobalType]>>
}

impl GlobalType {
    pub fn simple(kind: TypeKind) -> Self {
        GlobalType{kind, args: None}
    }

    pub fn from_legacy(ty: &Type) -> GlobalType {
        let kind = convert_legacy_ty(ty);
        GlobalType{ kind, args: None }
    }

    pub fn byte_size(&self) -> usize {
        match self.kind {
            TypeKind::Int(Some((IntWidth::IntSize,_))) => 8,
            TypeKind::Int(Some((IntWidth::Int8,_))) => 1,
            TypeKind::Int(Some((IntWidth::Int16,_))) => 2,
            TypeKind::Int(Some((IntWidth::Int32,_))) => 4,
            TypeKind::Int(Some((IntWidth::Int64,_))) => 8,
            TypeKind::Int(Some((IntWidth::Int128,_))) => 16,

            TypeKind::Float(Some(FloatWidth::Float32)) => 4,
            TypeKind::Float(Some(FloatWidth::Float64)) => 8,

            TypeKind::Bool => 1,
            TypeKind::Char => 4,
            
            TypeKind::Never => 0,
            TypeKind::Tuple => {
                assert!(self.args.is_none());
                0
            },
            _ => panic!("todo size {:?}",self.kind)
        }
    }

    pub fn byte_align(&self) -> usize {
        match self.kind {
            TypeKind::Int(Some((IntWidth::IntSize,_))) => 8,
            TypeKind::Int(Some((IntWidth::Int8,_))) => 1,
            TypeKind::Int(Some((IntWidth::Int16,_))) => 2,
            TypeKind::Int(Some((IntWidth::Int32,_))) => 4,
            TypeKind::Int(Some((IntWidth::Int64,_))) => 8,
            TypeKind::Int(Some((IntWidth::Int128,_))) => 16,

            TypeKind::Float(Some(FloatWidth::Float32)) => 4,
            TypeKind::Float(Some(FloatWidth::Float64)) => 8,

            TypeKind::Bool => 1,
            TypeKind::Char => 4,

            TypeKind::Never => 1,
            TypeKind::Tuple => {
                assert!(self.args.is_none());
                1
            },
            _ => panic!("todo align {:?}",self.kind)
        }
    }

    pub fn is_signed(&self) -> bool {
        if let TypeKind::Int(Some((_,IntSign::Signed))) = self.kind {
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
enum TypeConstraint {
    Equal(TypeVar,TypeVar),
    EqualLit(TypeVar,GlobalType),
    Assign{src: TypeVar, dst: TypeVar},
    OpBinary{op: OpBinary, res: TypeVar, lhs: TypeVar, rhs: TypeVar},
    OpUnary{op: OpUnary, res: TypeVar, arg: TypeVar},
    CheckBlockNever{var: TypeVar, expr_id: u32},
    LeastUpperBound2{res: TypeVar, arg1: TypeVar, arg2: TypeVar}
}

impl TypeConstraint {
    // used for conditions and &&/|| args.
    // todo possibly allow never?
    fn bool(var: TypeVar) -> Self {
        TypeConstraint::EqualLit(var,GlobalType::simple(TypeKind::Bool))
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

pub struct FuncTypes {
    var_types: Vec<LocalType>,
    expr_vars: Vec<TypeVar>,
    constraints: Vec<TypeConstraint>,
    constraints_sat: Vec<TypeConstraint>
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
            constraints_sat: vec!()
        };

        for i in 0..arg_count {
            let var_index = func.vars[i];
            types.init_expr(func, var_index);
        }

        let main_var = types.init_expr(func, func.root_expr as u32);
        types.add_constraint(TypeConstraint::EqualLit(main_var, func.ret_ty.clone()));

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
                let kind = convert_legacy_ty(ty);
                self.new_var(kind)
            }
            Expr::CastPrimitive(arg, ty) => {
                self.init_expr(func,*arg);

                let kind = convert_legacy_ty(ty);
                self.new_var(kind)
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

            Expr::CallBuiltin(name, args) => {
                let entry = BUILTINS.get(name.as_str());
                if entry.is_none() {
                    panic!("invalid builtin: {}", name);
                }

                let sig = &entry.unwrap().1;

                assert_eq!(args.len(),sig.inputs.len());

                for (arg,ty) in args.iter().zip(sig.inputs.iter()) {
                    let var = self.init_expr(func,*arg);
                    self.add_constraint(TypeConstraint::EqualLit(var,GlobalType::from_legacy(ty)));
                }

                self.new_var(convert_legacy_ty(&sig.output))
            }
            Expr::Call(call_func, args) => {
                let sig = call_func.sig();

                assert_eq!(args.len(),sig.inputs.len());

                for (arg,ty) in args.iter().zip(sig.inputs.iter()) {
                    let var = self.init_expr(func,*arg);
                    self.add_constraint(TypeConstraint::EqualLit(var,GlobalType::from_legacy(ty)));
                }

                self.new_var(convert_legacy_ty(&sig.output))
            }
            Expr::Return(ret_expr) => {
                let ret_var = if let Some(ret_expr) = ret_expr {
                    self.init_expr(func,*ret_expr)
                } else {
                    TYPE_VOID
                };

                // todo coerce?
                self.add_constraint(TypeConstraint::EqualLit(ret_var,func.ret_ty.clone()));

                self.new_var(TypeKind::Never)
            }
            Expr::Break(_, break_expr) => {
                assert!(break_expr.is_none());

                self.new_var(TypeKind::Never)
            }
            Expr::Continue(_) => {
                self.new_var(TypeKind::Never)
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

    pub fn solve(&mut self, func: &mut FuncHIR) {
        let constraints = std::mem::take(&mut self.constraints);
        for c in constraints {
            let sat = match &c {
                TypeConstraint::Equal(var1, var2) => self.solve_equal(*var1,*var2),
                TypeConstraint::EqualLit(var, ty) => {

                    let lt = &mut self.var_types[var.0 as usize];

                    if lt.kind == TypeKind::Never {
                        // dumb hack -- todo probably replace this constraint entirely with some coerceto constraint
                        true
                    } else {
                        lt.unify_g(&ty);
                
                        lt.kind.is_known()
                    }
                }
                TypeConstraint::Assign { src, dst } => self.solve_assign(*src,*dst),
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
        assert_eq!(ty.args.len(),0);
        GlobalType::simple(ty.kind.clone())
    }

    fn solve_equal(&mut self, var1: TypeVar, var2: TypeVar) -> bool {
        if var1 == var2 {
            return true;
            //return self.get(var1).kind.is_known();
        }

        let (t1,t2) = self.get_mut_2(var1, var2);

        if t1 != t2 {
            t1.unify(t2);
        }

        t1.kind.is_known()
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

        self.get(src_var).kind.is_known() && self.get(dst_var).kind.is_known()
    }

    fn solve_lub2(&mut self, res_var: TypeVar, arg1_var: TypeVar, arg2_var: TypeVar) -> bool {

        // if either argument is of a known type that cannot coerce to another type, assume it is our result type
        if self.get(arg1_var).kind.cannot_coerce() {
            self.solve_equal(arg1_var, res_var);
            return self.get(arg1_var).kind.is_known() && self.get(arg2_var).kind.is_known();
        }
        if self.get(arg2_var).kind.cannot_coerce() {
            self.solve_equal(arg2_var, res_var);
            return self.get(arg1_var).kind.is_known() && self.get(arg2_var).kind.is_known();
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

                    self.get(lhs).kind.is_known()
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
        if ty.kind.is_known() {
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
        let goal_ty = GlobalType::simple(if is_never { TypeKind::Never } else { TypeKind::Tuple });

        let current_ty = &mut self.var_types[var.0 as usize];

        current_ty.unify_g(&goal_ty);

        return true;
    }

    fn check_expr_is_never(&self, func: &FuncHIR, expr_id: u32) -> Option<bool> {
        let expr = &func.exprs[expr_id as usize].expr;
        match expr {

            // check children
            Expr::UnOp(arg,_) => {
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
            Expr::Block(_) => {
                let res_var = self.expr_vars[expr_id as usize];
                self.check_ty_never(res_var)
            }

            // never diverge
            Expr::While(..) | // <- looks like the cond can just be ignored
            Expr::DeclVar(_) | Expr::Var(..) | Expr::CastPrimitive(..) |
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
        assert_eq!(ty.args.len(),0);
        match ty.kind {
            TypeKind::Never => Some(true),
            TypeKind::Unknown => None,
            _ => Some(false)
        }
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
