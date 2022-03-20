// based loosly on https://github.com/bytecodealliance/cranelift-jit-demo/blob/main/src/jit.rs

use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr;
use std::time::Instant;

use cranelift::codegen::ir::StackSlot;
use cranelift::codegen::Context;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::prelude::{
    isa::CallConv, types, AbiParam, EntityRef, InstBuilder, IntCC, Value, Variable,
};
use cranelift::prelude::{FloatCC, MemFlags, StackSlotData, StackSlotKind, TrapCode};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, FuncId, Linkage, Module};
use syn::{BinOp, UnOp};

use crate::builtin::BUILTINS;
use crate::disassemble::disassemble;
use crate::hir::func::{Block, Expr, ExprInfo, FuncHIR};
use crate::hir::item::Function;
use crate::hir::types::{ComplexType, FloatType, IntType, Signature, Type};
use crate::hir::var_storage::{get_var_storage, VarStorage};
use crate::PTR_WIDTH;
use crate::profiler::profile;

#[derive(Debug)]
enum CType {
    Void,
    Never,
    Scalar(cranelift::prelude::Type),
}

impl CType {
    fn unwrap_scalar(&self) -> cranelift::prelude::Type {
        if let CType::Scalar(val) = self {
            *val
        } else {
            panic!("failed to unwrap scalar type");
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CVal {
    Void,
    Scalar(Value),
}

impl CVal {
    fn unwrap_scalar(&self) -> Value {
        if let CVal::Scalar(val) = self {
            *val
        } else {
            panic!("failed to unwrap scalar value");
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CPlace {
    Register(Variable),
    Stack(StackSlot, i32),
    Pointer(Value),
    None // place used for ZSTs
}

type CValOrNever = Result<CVal, ()>;
type CPlaceOrNever = Result<CPlace, ()>;

type CSignature = cranelift::prelude::Signature;

fn ptr_ty() -> cranelift::prelude::Type {
    assert!(PTR_WIDTH == 8);
    types::I64
}

pub extern "C" fn jit_compile(func: &Function) -> *const u8 {
    let start = Instant::now();
    let result = JIT_CONTEXT.with(|rc| {
        let mut jit = rc.borrow_mut();
        jit.compile(func)
    });
    if crate::LOG_JITS {
        println!("JIT: {} {:?}", func.debug_name, start.elapsed());
    }

    match result {
        Ok(ptr) => {
            func.c_fn.set(ptr);

            ptr
        }
        Err(msg) => panic!("jit error: {}", msg),
    }
}

thread_local! {
    static JIT_CONTEXT: RefCell<JIT> = Default::default();
}

struct JIT {
    module: JITModule,
    ctx: Context,
    builder_ctx: FunctionBuilderContext,
    data_ctx: DataContext,
    builtins: HashMap<String, FuncId>,
}

impl Default for JIT {
    fn default() -> Self {
        let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names());

        for (key, (ptr, _)) in BUILTINS.iter() {
            let name = format!("builtin_{}", key);
            jit_builder.symbol(name, *ptr as *const u8);
        }

        let mut builtins = HashMap::new();
        let mut module = JITModule::new(jit_builder);
        for (key, (_, sig)) in BUILTINS.iter() {
            let name = format!("builtin_{}", key);
            let csig = lower_sig(sig);
            let fn_id = module
                .declare_function(&name, Linkage::Import, &csig)
                .expect("failed to declare builtin");
            builtins.insert(String::from(*key), fn_id);
        }

        let ctx = module.make_context();
        let builder_ctx = FunctionBuilderContext::new();
        let data_ctx = DataContext::new();

        //let fn_id = self.module.declare_function("func", Linkage::Export, &self.ctx.func.signature)

        Self {
            module,
            ctx,
            builder_ctx,
            data_ctx,
            builtins,
        }
    }
}

impl JIT {
    fn compile(&mut self, func: &Function) -> Result<*const u8, String> {
        let sig = func.sig();
        let ir = func.hir();

        let var_kinds = get_var_storage(ir);

        self.ctx.func.signature = lower_sig(sig);

        let fn_name = format!("skitter_{}", func as *const Function as usize);
        let fn_id = self
            .module
            .declare_function(&fn_name, Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| e.to_string())?;

        profile("lower HIR -> CLIF",|| {
            let mut jit_func = JITFunc {
                input_fn: ir,
                fn_builder: FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx),
                module: &self.module,
                builtins: &self.builtins,
                var_storage: vec![],
                active_loops: vec![],
            };

            jit_func.compile(var_kinds);
        });
        if crate::VERBOSE {
            println!("CLIF IR ===============>\n{}", self.ctx.func);
        }

        let compiled_fn = profile("codegen",|| {
            self
                .module
                .define_function(fn_id, &mut self.ctx)
                .map_err(|e| e.to_string())
        })?;
        
        let size = compiled_fn.size as usize;

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions();

        let code = self.module.get_finalized_function(fn_id);

        if crate::VERBOSE {
            let compiled_slice = unsafe { std::slice::from_raw_parts(code, size) };
            disassemble(compiled_slice);
        }

        Ok(code)
    }
}

fn lower_type(ty: Type) -> CType {
    match ty {
        // Ints
        Type::Int(IntType::I128) | Type::Int(IntType::U128) => {
            panic!("128 bit integers are broken")
        }
        Type::Int(IntType::I64) | Type::Int(IntType::U64) => CType::Scalar(types::I64),
        Type::Int(IntType::I32) | Type::Int(IntType::U32) => CType::Scalar(types::I32),
        Type::Int(IntType::I16) | Type::Int(IntType::U16) => CType::Scalar(types::I16),
        Type::Int(IntType::I8) | Type::Int(IntType::U8) => CType::Scalar(types::I8),
        Type::Int(IntType::ISize) | Type::Int(IntType::USize) => CType::Scalar(ptr_ty()),

        // Refs
        Type::Complex(ComplexType::Ref(..)) => CType::Scalar(ptr_ty()),

        // Floats
        Type::Float(FloatType::F64) => CType::Scalar(types::F64),
        Type::Float(FloatType::F32) => CType::Scalar(types::F32),

        Type::Bool => CType::Scalar(types::B1),
        Type::Char => CType::Scalar(types::I32),
        Type::Void => CType::Void,
        Type::Never => CType::Never,

        _ => panic!("unknown type {:?}", ty),
    }
}

fn lower_sig(sig: &Signature) -> CSignature {
    #[cfg(not(target_os = "windows"))]
    let call_conv = CallConv::SystemV;

    let mut result = CSignature::new(call_conv);

    for ty in &sig.inputs {
        if let CType::Scalar(cty) = lower_type(*ty) {
            result.params.push(AbiParam::new(cty));
        }
    }

    if let CType::Scalar(cty) = lower_type(sig.output) {
        result.returns.push(AbiParam::new(cty));
    }

    result
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum LowBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,

    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,

    BitAnd,
    BitOr,
    BitXor,

    BitShiftLeft,
    BitShiftRight,

    LogicAnd,
    LogicOr,
}

impl LowBinOp {
    fn is_compare(&self) -> bool {
        match self {
            LowBinOp::Gt
            | LowBinOp::Lt
            | LowBinOp::Ge
            | LowBinOp::Le
            | LowBinOp::Eq
            | LowBinOp::Ne => true,
            _ => false,
        }
    }

    fn int_cond_code(&self, sign: bool) -> IntCC {
        match (self, sign) {
            (LowBinOp::Gt, true) => IntCC::SignedGreaterThan,
            (LowBinOp::Lt, true) => IntCC::SignedLessThan,
            (LowBinOp::Ge, true) => IntCC::SignedGreaterThanOrEqual,
            (LowBinOp::Le, true) => IntCC::SignedLessThanOrEqual,

            (LowBinOp::Gt, false) => IntCC::UnsignedGreaterThan,
            (LowBinOp::Lt, false) => IntCC::UnsignedLessThan,
            (LowBinOp::Ge, false) => IntCC::UnsignedGreaterThanOrEqual,
            (LowBinOp::Le, false) => IntCC::UnsignedLessThanOrEqual,

            (LowBinOp::Eq, _) => IntCC::Equal,
            (LowBinOp::Ne, _) => IntCC::NotEqual,

            _ => panic!("int cond code for {:?} {}", self, sign),
        }
    }

    fn float_cond_code(&self) -> FloatCC {
        match self {
            LowBinOp::Gt => FloatCC::GreaterThan,
            LowBinOp::Lt => FloatCC::LessThan,
            LowBinOp::Ge => FloatCC::GreaterThanOrEqual,
            LowBinOp::Le => FloatCC::LessThanOrEqual,

            LowBinOp::Eq => FloatCC::Equal,
            LowBinOp::Ne => FloatCC::NotEqual,
            _ => panic!("float cond code for {:?}", self),
        }
    }
}

// Second parameter indicates whether the op is an assignment
fn lower_bin_op(op: &BinOp) -> (LowBinOp, bool) {
    match op {
        BinOp::Add(_) => (LowBinOp::Add, false),
        BinOp::Sub(_) => (LowBinOp::Sub, false),
        BinOp::Mul(_) => (LowBinOp::Mul, false),
        BinOp::Div(_) => (LowBinOp::Div, false),
        BinOp::Rem(_) => (LowBinOp::Rem, false),

        BinOp::AddEq(_) => (LowBinOp::Add, true),
        BinOp::SubEq(_) => (LowBinOp::Sub, true),
        BinOp::MulEq(_) => (LowBinOp::Mul, true),
        BinOp::DivEq(_) => (LowBinOp::Div, true),
        BinOp::RemEq(_) => (LowBinOp::Rem, true),

        BinOp::Gt(_) => (LowBinOp::Gt, false),
        BinOp::Lt(_) => (LowBinOp::Lt, false),
        BinOp::Ge(_) => (LowBinOp::Ge, false),
        BinOp::Le(_) => (LowBinOp::Le, false),
        BinOp::Eq(_) => (LowBinOp::Eq, false),
        BinOp::Ne(_) => (LowBinOp::Ne, false),

        BinOp::BitAnd(_) => (LowBinOp::BitAnd, false),
        BinOp::BitOr(_) => (LowBinOp::BitOr, false),
        BinOp::BitXor(_) => (LowBinOp::BitXor, false),

        BinOp::BitAndEq(_) => (LowBinOp::BitAnd, true),
        BinOp::BitOrEq(_) => (LowBinOp::BitOr, true),
        BinOp::BitXorEq(_) => (LowBinOp::BitXor, true),

        BinOp::Shl(_) => (LowBinOp::BitShiftLeft, false),
        BinOp::Shr(_) => (LowBinOp::BitShiftRight, false),

        BinOp::ShlEq(_) => (LowBinOp::BitShiftLeft, true),
        BinOp::ShrEq(_) => (LowBinOp::BitShiftRight, true),

        BinOp::And(_) => (LowBinOp::LogicAnd, false),
        BinOp::Or(_) => (LowBinOp::LogicOr, false)
    }
}

struct LoopBlocks {
    loop_id: u32,
    b_break: cranelift::prelude::Block,
    b_continue: cranelift::prelude::Block,
}

struct JITFunc<'a> {
    input_fn: &'a FuncHIR,
    fn_builder: FunctionBuilder<'a>,
    module: &'a JITModule,
    builtins: &'a HashMap<String, FuncId>,
    var_storage: Vec<CPlace>,
    active_loops: Vec<LoopBlocks>,
}

impl<'a> JITFunc<'a> {
    pub fn compile(&mut self, var_kinds: Vec<VarStorage>) {
        let entry_block = self.fn_builder.create_block();

        let mut next_var_id = 0;

        self.var_storage = self
            .input_fn
            .vars
            .iter()
            .zip(var_kinds)
            .map(|(expr_index, storage)| {
                let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[*expr_index as usize];
                if let Expr::Var(_) = expr {
                    match storage {
                        VarStorage::Register => {
                            if let CType::Scalar(cty) = lower_type(*ty) {
                                let var = Variable::new(next_var_id);
                                next_var_id += 1;
                                self.fn_builder.declare_var(var, cty);
                                CPlace::Register(var)
                            } else {
                                panic!("attempt to alloc register for {:?}", ty);
                            }
                        }
                        VarStorage::Stack => {
                            let size = ty.byte_size();
                            let slot = self.fn_builder.create_stack_slot(StackSlotData::new(
                                StackSlotKind::ExplicitSlot,
                                size as u32,
                            ));
                            CPlace::Stack(slot, 0)
                        }
                        VarStorage::None => CPlace::None,
                        VarStorage::Pointer => panic!("init stack pointer"), // ssa value
                    }
                } else {
                    panic!("can not create var from {:?}", expr);
                }
            })
            .collect();

        self.fn_builder
            .append_block_params_for_function_params(entry_block);
        self.fn_builder.switch_to_block(entry_block);
        self.fn_builder.seal_block(entry_block);

        let param_vals = Vec::from(self.fn_builder.block_params(entry_block));
        for (var_id, val) in param_vals.iter().enumerate() {
            let var = Variable::new(var_id);
            self.fn_builder.def_var(var, *val);
        }

        let res = self.lower_expr(self.input_fn.root_expr as u32);
        match res {
            Ok(CVal::Scalar(val)) => {
                self.fn_builder.ins().return_(&[val]);
            }
            Ok(CVal::Void) => {
                self.fn_builder.ins().return_(&[]);
            }
            Err(_) => (),
            _ => panic!("can not return {:?}", res),
        }

        self.fn_builder.finalize();
    }

    fn lower_expr(&mut self, expr_id: u32) -> CValOrNever {
        let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[expr_id as usize];
        let cty = lower_type(*ty);
        Ok(match expr {
            Expr::Var(var_id) => match &self.var_storage[*var_id as usize] {
                CPlace::Register(reg) => CVal::Scalar(self.fn_builder.use_var(*reg)),
                vs => panic!("uses var {:?}", vs),
            },
            Expr::CastPrimitive(arg) => {
                let src_ty = self.input_fn.exprs[*arg as usize].ty;

                let arg = self.lower_expr(*arg)?.unwrap_scalar();

                if (src_ty.is_int() || src_ty == Type::Char) && ty.is_int() {
                    let size_src = src_ty.byte_size();
                    let size_dest = ty.byte_size();

                    if size_src == size_dest {
                        CVal::Scalar(arg)
                    } else if size_src < size_dest {
                        // widening: our type of extension is determined by the source type
                        if src_ty.is_signed() {
                            CVal::Scalar(self.fn_builder.ins().sextend(cty.unwrap_scalar(), arg))
                        } else {
                            CVal::Scalar(self.fn_builder.ins().uextend(cty.unwrap_scalar(), arg))
                        }
                    } else {
                        // narrowing
                        CVal::Scalar(self.fn_builder.ins().ireduce(cty.unwrap_scalar(), arg))
                    }
                } else if src_ty == Type::Int(IntType::U8) && *ty == Type::Char {
                    CVal::Scalar(self.fn_builder.ins().uextend(types::I32, arg))
                } else {
                    panic!("non integer casts nyi");
                }
            }
            Expr::Assign(dest, src) => {
                let dest_place = self.lower_place(*dest)?;
                let src_val = self.lower_expr(*src)?;

                self.lower_assign(dest_place, src_val, *ty);
                CVal::Void
            }
            Expr::BinOpPrimitive(lhs, op, rhs) => {
                let (op, is_assign) = lower_bin_op(op);

                if op == LowBinOp::LogicAnd || op == LowBinOp::LogicOr {
                    let cond = self.lower_expr(*lhs)?.unwrap_scalar();

                    let right_cb = self.fn_builder.create_block();
                    let final_cb = self.fn_builder.create_block();

                    if op == LowBinOp::LogicAnd {
                        self.fn_builder.ins().brnz(cond, right_cb, &[]);
                    } else {
                        self.fn_builder.ins().brz(cond, right_cb, &[]);
                    }
                    self.fn_builder.ins().jump(final_cb, &[cond]);

                    self.fn_builder.seal_block(right_cb);
                    self.fn_builder.switch_to_block(right_cb);

                    if let Ok(rval) = self.lower_expr(*rhs) {
                        self.fn_builder
                            .ins()
                            .jump(final_cb, &[rval.unwrap_scalar()]);
                    }

                    self.fn_builder.seal_block(final_cb);
                    self.fn_builder.switch_to_block(final_cb);

                    self.fn_builder.append_block_param(final_cb, types::B1);
                    return Ok(CVal::Scalar(self.fn_builder.block_params(final_cb)[0]));
                }

                let lval = self.lower_expr(*lhs)?.unwrap_scalar();
                let rval = self.lower_expr(*rhs)?.unwrap_scalar();

                let arg_ty = self.input_fn.exprs[*lhs as usize].ty;
                let res_val = CVal::Scalar(match (arg_ty, op) {
                    (Type::Int(_), LowBinOp::Add) => self.fn_builder.ins().iadd(lval, rval),
                    (Type::Int(_), LowBinOp::Sub) => self.fn_builder.ins().isub(lval, rval),
                    (Type::Int(_), LowBinOp::Mul) => self.fn_builder.ins().imul(lval, rval),
                    (Type::Int(_), LowBinOp::Div) => {
                        if ty.is_signed() {
                            self.fn_builder.ins().sdiv(lval, rval)
                        } else {
                            self.fn_builder.ins().udiv(lval, rval)
                        }
                    }
                    (Type::Int(_), LowBinOp::Rem) => {
                        if ty.is_signed() {
                            self.fn_builder.ins().srem(lval, rval)
                        } else {
                            self.fn_builder.ins().urem(lval, rval)
                        }
                    }
                    (Type::Int(_), LowBinOp::BitShiftLeft) => {
                        self.fn_builder.ins().ishl(lval, rval)
                    }
                    (Type::Int(_), LowBinOp::BitShiftRight) => {
                        if ty.is_signed() {
                            self.fn_builder.ins().sshr(lval, rval)
                        } else {
                            self.fn_builder.ins().ushr(lval, rval)
                        }
                    }
                    (_, LowBinOp::BitAnd) => self.fn_builder.ins().band(lval, rval),
                    (_, LowBinOp::BitOr) => self.fn_builder.ins().bor(lval, rval),
                    (_, LowBinOp::BitXor) => self.fn_builder.ins().bxor(lval, rval),

                    (Type::Float(_), LowBinOp::Add) => self.fn_builder.ins().fadd(lval, rval),
                    (Type::Float(_), LowBinOp::Sub) => self.fn_builder.ins().fsub(lval, rval),
                    (Type::Float(_), LowBinOp::Mul) => self.fn_builder.ins().fmul(lval, rval),
                    (Type::Float(_), LowBinOp::Div) => self.fn_builder.ins().fdiv(lval, rval),
                    (Type::Float(_), LowBinOp::Rem) => {
                        // why use 1 instruction when 4 do trick
                        // x - (x / y).trunc() * y
                        let t1 = self.fn_builder.ins().fdiv(lval, rval);
                        let t2 = self.fn_builder.ins().trunc(t1);
                        let t3 = self.fn_builder.ins().fmul(t2, rval);
                        self.fn_builder.ins().fsub(lval, t3)
                    }
                    _ if op.is_compare() => {
                        match arg_ty {
                            Type::Int(_) | Type::Char => self.fn_builder.ins().icmp(
                                op.int_cond_code(arg_ty.is_signed()),
                                lval,
                                rval,
                            ),
                            Type::Float(_) => {
                                self.fn_builder.ins().fcmp(op.float_cond_code(), lval, rval)
                            }
                            Type::Bool => {
                                if op == LowBinOp::Eq {
                                    let tmp = self.fn_builder.ins().bxor(lval, rval);
                                    self.fn_builder.ins().bnot(tmp)
                                } else if op == LowBinOp::Ne {
                                    self.fn_builder.ins().bxor(lval, rval)
                                } else {
                                    panic!("bad primitive bool compare {:?}", op);
                                }
                            }
                            _ => panic!("can't compare {:?}", arg_ty),
                        }
                    }

                    _ => panic!("can't compile primitive op {:?} {:?}", ty, op),
                });

                if is_assign {
                    let dest = self.lower_place(*lhs).unwrap();
                    self.lower_assign(dest, res_val, *ty);
                    CVal::Void
                } else {
                    res_val
                }
            }
            Expr::UnOpPrimitive(arg, op) => {
                let arg = self.lower_expr(*arg)?.unwrap_scalar();

                CVal::Scalar(match *op {
                    UnOp::Neg(_) => {
                        if ty.is_int() {
                            self.fn_builder.ins().ineg(arg)
                        } else {
                            self.fn_builder.ins().fneg(arg)
                        }
                    }
                    UnOp::Not(_) => self.fn_builder.ins().bnot(arg),
                    _ => panic!("todo op {:?}", op),
                })
            }

            // Literals, should be mostly good.
            Expr::LitInt(x) => {
                // TODO we assume the int is register-sized
                let x: Result<i64, _> = (*x).try_into();
                if let Ok(n) = x {
                    CVal::Scalar(self.fn_builder.ins().iconst(cty.unwrap_scalar(), n))
                } else {
                    panic!("int too wide");
                }
            }
            Expr::LitFloat(x) => CVal::Scalar(match ty {
                Type::Float(FloatType::F64) => self.fn_builder.ins().f64const(*x),
                Type::Float(FloatType::F32) => self.fn_builder.ins().f32const(*x as f32),
                _ => panic!(),
            }),
            Expr::LitChar(x) => {
                let x = *x as i64;
                CVal::Scalar(self.fn_builder.ins().iconst(cty.unwrap_scalar(), x))
            }
            Expr::LitBool(x) => CVal::Scalar(self.fn_builder.ins().bconst(cty.unwrap_scalar(), *x)),
            Expr::LitVoid => CVal::Void,

            Expr::Ref(x, _is_mut) => {
                let place = self.lower_place(*x)?;
                match place {
                    CPlace::Stack(slot, offset) => {
                        CVal::Scalar(self.fn_builder.ins().stack_addr(ptr_ty(), slot, offset))
                    }
                    CPlace::Pointer(ptr) => CVal::Scalar(ptr),
                    _ => panic!("attempt to ref {:?}", place),
                }
            }
            Expr::DeRef(x) => {
                let addr = self.lower_expr(*x)?.unwrap_scalar();
                match cty {
                    CType::Scalar(vt) => {
                        CVal::Scalar(self.fn_builder.ins().load(vt, MemFlags::trusted(), addr, 0))
                    }
                    _ => panic!("cannot deref {:?}", cty),
                }
            }
            Expr::Block(block) => self.lower_block(block)?,
            // if-then's without an else
            Expr::If(cond, then_block, None) => {
                let cond = self.lower_expr(*cond)?.unwrap_scalar();

                let then_cb = self.fn_builder.create_block();
                let final_cb = self.fn_builder.create_block();

                self.fn_builder.ins().brnz(cond, then_cb, &[]);
                self.fn_builder.ins().jump(final_cb, &[]);

                self.fn_builder.seal_block(then_cb);
                self.fn_builder.switch_to_block(then_cb);

                if let Ok(_) = self.lower_block(then_block) {
                    self.fn_builder.ins().jump(final_cb, &[]);
                }

                self.fn_builder.seal_block(final_cb);
                self.fn_builder.switch_to_block(final_cb);

                CVal::Void
            }
            // if-then-else which can yield values
            Expr::If(cond, then_block, Some(else_expr)) => {
                let cond = self.lower_expr(*cond)?.unwrap_scalar();

                let then_cb = self.fn_builder.create_block();
                let else_cb = self.fn_builder.create_block();
                let final_cb = self.fn_builder.create_block();

                self.fn_builder.ins().brnz(cond, then_cb, &[]);
                self.fn_builder.ins().jump(else_cb, &[]);

                self.fn_builder.seal_block(then_cb);
                self.fn_builder.seal_block(else_cb);

                // then branch
                self.fn_builder.switch_to_block(then_cb);

                let then_res = self.lower_block(then_block);
                match then_res {
                    Ok(CVal::Scalar(v)) => {
                        self.fn_builder.ins().jump(final_cb, &[v]);
                    }
                    Ok(CVal::Void) => {
                        self.fn_builder.ins().jump(final_cb, &[]);
                    }
                    Err(_) => (),
                }

                // else branch
                self.fn_builder.switch_to_block(else_cb);

                let else_res = self.lower_expr(*else_expr);
                match else_res {
                    Ok(CVal::Scalar(v)) => {
                        self.fn_builder.ins().jump(final_cb, &[v]);
                    }
                    Ok(CVal::Void) => {
                        self.fn_builder.ins().jump(final_cb, &[]);
                    }
                    Err(_) => (),
                }

                // if both branches are never, we are also never
                if then_res.is_err() && else_res.is_err() {
                    return Err(());
                }

                // final, unified block
                self.fn_builder.switch_to_block(final_cb);
                self.fn_builder.seal_block(final_cb);

                let res = if let CType::Scalar(cty) = cty {
                    self.fn_builder.append_block_param(final_cb, cty);
                    CVal::Scalar(self.fn_builder.block_params(final_cb)[0])
                } else {
                    CVal::Void
                };

                res
            }
            Expr::While(cond, body_block) => {
                let cond_cb = self.fn_builder.create_block();

                self.fn_builder.ins().jump(cond_cb, &[]);

                // cond block
                self.fn_builder.switch_to_block(cond_cb);

                let cond = self.lower_expr(*cond)?.unwrap_scalar();

                let body_cb = self.fn_builder.create_block();
                let final_cb = self.fn_builder.create_block();

                self.fn_builder.ins().brnz(cond, body_cb, &[]);
                self.fn_builder.ins().jump(final_cb, &[]);

                self.fn_builder.seal_block(body_cb);

                // body block
                self.active_loops.push(LoopBlocks {
                    loop_id: expr_id,
                    b_break: final_cb,
                    b_continue: cond_cb,
                });
                self.fn_builder.switch_to_block(body_cb);
                if let Ok(_) = self.lower_block(body_block) {
                    self.fn_builder.ins().jump(cond_cb, &[]);
                }
                {
                    let popped_loop_info = self.active_loops.pop().unwrap();
                    assert_eq!(popped_loop_info.loop_id, expr_id);
                }

                // must be sealed after any potential breaks or continues
                self.fn_builder.seal_block(cond_cb);
                self.fn_builder.seal_block(final_cb);

                // switch to final block
                self.fn_builder.switch_to_block(final_cb);

                CVal::Void
            }
            Expr::Loop(body_block) => {
                let body_cb = self.fn_builder.create_block();
                let final_cb = self.fn_builder.create_block();
                self.fn_builder.ins().jump(body_cb, &[]);

                // body block
                self.active_loops.push(LoopBlocks {
                    loop_id: expr_id,
                    b_break: final_cb,
                    b_continue: body_cb,
                });
                self.fn_builder.switch_to_block(body_cb);
                if let Ok(_) = self.lower_block(body_block) {
                    self.fn_builder.ins().jump(body_cb, &[]);
                }
                {
                    let popped_loop_info = self.active_loops.pop().unwrap();
                    assert_eq!(popped_loop_info.loop_id, expr_id);
                }

                // must be sealed after any potential breaks or continues
                self.fn_builder.seal_block(body_cb);
                self.fn_builder.seal_block(final_cb);

                // switch to final block
                self.fn_builder.switch_to_block(final_cb);

                let res = if let CType::Scalar(cty) = cty {
                    self.fn_builder.append_block_param(final_cb, cty);
                    CVal::Scalar(self.fn_builder.block_params(final_cb)[0])
                } else {
                    CVal::Void
                };

                res
            }
            Expr::Break(loop_id, val) => {
                let val = if let Some(val) = val {
                    self.lower_expr(*val)?
                } else {
                    CVal::Void
                };

                let loop_info = self
                    .active_loops
                    .iter()
                    .find(|x| x.loop_id == *loop_id)
                    .unwrap();
                match val {
                    CVal::Scalar(val) => {
                        self.fn_builder.ins().jump(loop_info.b_break, &[val]);
                    }
                    CVal::Void => {
                        self.fn_builder.ins().jump(loop_info.b_break, &[]);
                    }
                }

                return Err(());
            }
            Expr::Continue(loop_id) => {
                let loop_info = self
                    .active_loops
                    .iter()
                    .find(|x| x.loop_id == *loop_id)
                    .unwrap();
                self.fn_builder.ins().jump(loop_info.b_continue, &[]);

                return Err(());
            }
            Expr::CallBuiltin(name, args) => {
                let fn_id = self
                    .builtins
                    .get(name)
                    .expect("failed to lookup builtin id");
                let fn_ref = self
                    .module
                    .declare_func_in_func(*fn_id, self.fn_builder.func);

                let mut c_args = Vec::new();
                for arg in args {
                    if let CVal::Scalar(val) = self.lower_expr(*arg)? {
                        c_args.push(val);
                    }
                }

                let call_inst = self.fn_builder.ins().call(fn_ref, &c_args);

                let call_res = self.fn_builder.inst_results(call_inst);
                if call_res.len() > 0 {
                    CVal::Scalar(call_res[0])
                } else {
                    CVal::Void
                }
            }
            Expr::Call(func, args) => {
                let jit_cb = self.fn_builder.create_block();
                let next_cb = self.fn_builder.create_block();

                let sig = func.sig();
                let c_sig = lower_sig(sig);
                //let returns_never = c_sig.returns
                let sig_id = self.fn_builder.import_signature(c_sig);

                let data_ptr = *func as *const Function;
                let data_val = self.fn_builder.ins().iconst(ptr_ty(), data_ptr as i64);
                let fn_val = self
                    .fn_builder
                    .ins()
                    .load(ptr_ty(), MemFlags::trusted(), data_val, 0);

                self.fn_builder.ins().brz(fn_val, jit_cb, &[]);
                self.fn_builder.ins().jump(next_cb, &[fn_val]);

                // jit long-path
                {
                    self.fn_builder.seal_block(jit_cb);
                    self.fn_builder.switch_to_block(jit_cb);

                    let jit_id = self
                        .builtins
                        .get("jit_compile")
                        .expect("failed to lookup builtin id");
                    let jit_ref = self
                        .module
                        .declare_func_in_func(*jit_id, self.fn_builder.func);

                    let jit_inst = self.fn_builder.ins().call(jit_ref, &[data_val]);
                    let fn_val = self.fn_builder.inst_results(jit_inst)[0];

                    self.fn_builder.ins().jump(next_cb, &[fn_val]);
                }

                self.fn_builder.seal_block(next_cb);
                self.fn_builder.switch_to_block(next_cb);

                self.fn_builder.append_block_param(next_cb, ptr_ty());
                let fn_val = self.fn_builder.block_params(next_cb)[0];

                let mut c_args = Vec::new();
                for arg in args {
                    if let CVal::Scalar(val) = self.lower_expr(*arg)? {
                        c_args.push(val);
                    }
                }

                let call_inst = self.fn_builder.ins().call_indirect(sig_id, fn_val, &c_args);

                if sig.output == Type::Never {
                    self.fn_builder.ins().trap(TrapCode::UnreachableCodeReached);
                    return Err(());
                }

                let call_res = self.fn_builder.inst_results(call_inst);
                if call_res.len() > 0 {
                    CVal::Scalar(call_res[0])
                } else {
                    CVal::Void
                }
            }
            _ => panic!("todo lower expr {:?}", expr),
        })
    }

    fn lower_block(&mut self, block: &Block) -> CValOrNever {
        for expr_id in &block.stmts {
            self.lower_expr(*expr_id)?;
        }

        if let Some(expr_id) = &block.result {
            self.lower_expr(*expr_id)
        } else {
            Ok(CVal::Void)
        }
    }

    fn lower_assign(&mut self, dest: CPlace, src: CVal, ty: Type) {
        let cty = lower_type(ty);
        match src {
            CVal::Scalar(src) => match dest {
                CPlace::Register(reg) => {
                    self.fn_builder.def_var(reg, src);
                }
                CPlace::Stack(slot, offset) => {
                    self.fn_builder.ins().stack_store(src, slot, offset);
                }
                CPlace::Pointer(addr) => {
                    self.fn_builder
                        .ins()
                        .store(MemFlags::trusted(), src, addr, 0);
                }
                _ => panic!("todo lower assignment {:?}", dest),
            },
            CVal::Void => (),
        }
    }

    fn lower_place(&mut self, expr_id: u32) -> CPlaceOrNever {
        let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[expr_id as usize];
        Ok(match expr {
            Expr::Var(var_id) => self.var_storage[*var_id as usize],
            Expr::DeRef(arg) => {
                let arg = self.lower_expr(*arg)?;
                if let CVal::Scalar(x) = arg {
                    CPlace::Pointer(x)
                } else {
                    panic!("attempt to deref {:?}", arg);
                }
            }
            // store temporaries on the stack to convert them to places
            Expr::LitInt(_) | Expr::LitFloat(_) => {
                let src = self.lower_expr(expr_id)?;

                if let CVal::Scalar(src) = src {
                    let size = ty.byte_size();
                    let slot = self.fn_builder.create_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        size as u32,
                    ));
                    self.fn_builder.ins().stack_store(src, slot, 0);
                    CPlace::Stack(slot, 0)
                } else {
                    panic!("can't get address of non-scalar")
                }
            }
            _ => panic!("get addr: {:?}", expr),
        })
    }
}
