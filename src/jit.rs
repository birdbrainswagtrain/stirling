// based loosly on https://github.com/bytecodealliance/cranelift-jit-demo/blob/main/src/jit.rs

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

use cranelift::codegen::Context;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::prelude::{MemFlags, FloatCC};
use cranelift::prelude::{
    isa::CallConv, types, AbiParam, EntityRef, InstBuilder, IntCC, Value, Variable,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, FuncId, Linkage, Module};
use syn::{BinOp, UnOp};

use crate::builtin::BUILTINS;
use crate::disassemble::disassemble;
use crate::hir::func::{Block, Expr, ExprInfo, FuncHIR};
use crate::hir::item::Function;
use crate::hir::types::{Signature, Type, TypeFloat, TypeInt};
use crate::PTR_WIDTH;

enum CType{
    Never,
    Void,
    Value(cranelift::prelude::Type),
}

enum CValInner {
    Void,
    Value(Value)
}

type CSignature = cranelift::prelude::Signature;
type CVal = Result<CValInner,()>;

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
    println!("JIT: {} {:?}", func.debug_name, start.elapsed());

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
        self.ctx.func.signature = lower_sig(sig);

        let fn_name = format!("skitter_{}", func as *const Function as usize);
        let fn_id = self
            .module
            .declare_function(&fn_name, Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| e.to_string())?;

        {
            let mut jit_func = JITFunc {
                input_fn: ir,
                fn_builder: FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx),
                module: &self.module,
                builtins: &self.builtins,
            };

            jit_func.compile();
        }

        let compiled_fn = self
            .module
            .define_function(fn_id, &mut self.ctx)
            .map_err(|e| e.to_string())?;
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
        Type::Int(TypeInt::I128) | Type::Int(TypeInt::U128) => {
            panic!("128 bit integers are broken")
        }
        Type::Int(TypeInt::I64) | Type::Int(TypeInt::U64) => Some(types::I64),
        Type::Int(TypeInt::I32) | Type::Int(TypeInt::U32) => Some(types::I32),
        Type::Int(TypeInt::I16) | Type::Int(TypeInt::U16) => Some(types::I16),
        Type::Int(TypeInt::I8) | Type::Int(TypeInt::U8) => Some(types::I8),
        Type::Int(TypeInt::ISize) | Type::Int(TypeInt::USize) => Some(ptr_ty()),

        // Floats
        Type::Float(TypeFloat::F64) => Some(types::F64),
        Type::Float(TypeFloat::F32) => Some(types::F32),

        Type::Bool => Some(types::B1),
        Type::Char => Some(types::I32),
        Type::Void => None,
        _ => panic!("unknown type"),
    }
}

fn lower_sig(sig: &Signature) -> CSignature {
    #[cfg(not(target_os = "windows"))]
    let call_conv = CallConv::SystemV;

    let mut result = CSignature::new(call_conv);

    for ty in &sig.inputs {
        if let Some(cty) = lower_type(*ty) {
            result.params.push(AbiParam::new(cty));
        }
    }

    if let Some(cty) = lower_type(sig.output) {
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
            _ => panic!("float cond code for {:?}", self)
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
        BinOp::Or(_) => (LowBinOp::LogicOr, false),

        _ => panic!("can't lower op {:?}", op),
    }
}

struct JITFunc<'a> {
    input_fn: &'a FuncHIR,
    fn_builder: FunctionBuilder<'a>,
    module: &'a JITModule,
    builtins: &'a HashMap<String, FuncId>,
}

impl<'a> JITFunc<'a> {
    pub fn compile(&mut self) {
        let entry_block = self.fn_builder.create_block();

        for index in self.input_fn.vars.iter() {
            let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[*index as usize];
            if let Expr::Var(var_id) = expr {
                if let Some(cty) = lower_type(*ty) {
                    let var = Variable::new(*var_id as usize);
                    self.fn_builder.declare_var(var, cty);
                }
            } else {
                panic!("can not create var from {:?}", expr);
            }
        }

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
        if let Some(res) = res {
            self.fn_builder.ins().return_(&[res]);
        } else {
            self.fn_builder.ins().return_(&[]);
        }

        self.fn_builder.finalize();
    }

    fn lower_expr(&mut self, expr_id: u32) -> CVal {
        let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[expr_id as usize];
        let cty = lower_type(*ty);
        match expr {
            Expr::Var(var_id) => {
                let var = Variable::new(*var_id as usize);
                Some(self.fn_builder.use_var(var))
            }
            Expr::CastPrimitive(arg) => {
                let src_ty = self.input_fn.exprs[*arg as usize].ty;

                let arg = self.lower_expr(*arg).unwrap();

                if src_ty.is_int() && ty.is_int() {
                    let size_src = src_ty.byte_size();
                    let size_dest = ty.byte_size();

                    if size_src == size_dest {
                        Some(arg)
                    } else if size_src < size_dest {
                        // widening: our type of extension is determined by the source type
                        if src_ty.is_signed() {
                            Some(self.fn_builder.ins().sextend(cty.unwrap(), arg))
                        } else {
                            Some(self.fn_builder.ins().uextend(cty.unwrap(), arg))
                        }
                    } else {
                        // narrowing
                        Some(self.fn_builder.ins().ireduce(cty.unwrap(), arg))
                    }
                } else {
                    panic!("non integer casts nyi");
                }
            }
            Expr::Assign(dest, src) => {
                let src_val = self.lower_expr(*src);
                self.lower_assign(*dest, src_val)
            }
            Expr::BinOpPrimitive(lhs, op, rhs) => {
                let (op, is_assign) = lower_bin_op(op);

                if op == LowBinOp::LogicAnd || op == LowBinOp::LogicOr {
                    let right_cb = self.fn_builder.create_block();
                    let final_cb = self.fn_builder.create_block();

                    let cond = self.lower_expr(*lhs).unwrap();

                    if op == LowBinOp::LogicAnd {
                        self.fn_builder.ins().brnz(cond, right_cb, &[]);
                    } else {
                        self.fn_builder.ins().brz(cond, right_cb, &[]);
                    }
                    self.fn_builder.ins().jump(final_cb, &[cond]);

                    self.fn_builder.seal_block(right_cb);
                    self.fn_builder.switch_to_block(right_cb);

                    let rval = self.lower_expr(*rhs).unwrap();

                    self.fn_builder.ins().jump(final_cb, &[rval]);

                    self.fn_builder.seal_block(final_cb);
                    self.fn_builder.switch_to_block(final_cb);

                    self.fn_builder.append_block_param(final_cb, types::B1);
                    return Some(self.fn_builder.block_params(final_cb)[0]);
                }

                let lval = self.lower_expr(*lhs).unwrap();
                let rval = self.lower_expr(*rhs).unwrap();

                let res_val = Some(match (ty, op) {
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
                        let arg_ty = self.input_fn.exprs[*lhs as usize].ty;
                        match arg_ty {
                            Type::Int(_) | Type::Char => self.fn_builder.ins().icmp(
                                op.int_cond_code(arg_ty.is_signed()),
                                lval,
                                rval,
                            ),
                            Type::Float(_) => self.fn_builder.ins().fcmp(
                                op.float_cond_code(),
                                lval,
                                rval,
                            ),
                            Type::Bool => {
                                if op == LowBinOp::Eq {
                                    let tmp = self.fn_builder.ins().bxor(lval, rval);
                                    self.fn_builder.ins().bnot(tmp)
                                } else if op == LowBinOp::Ne {
                                    self.fn_builder.ins().bxor(lval, rval)
                                } else {
                                    panic!("bad primitive bool compare {:?}", op);
                                }
                            },
                            _ => panic!("can't compare {:?}", arg_ty),
                        }
                    }

                    _ => panic!("can't compile primitive op {:?} {:?}", ty, op),
                });

                if is_assign {
                    self.lower_assign(*lhs, res_val);
                }

                res_val
            }
            Expr::UnOpPrimitive(arg, op) => {
                let arg = self.lower_expr(*arg).unwrap();

                Some(match *op {
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
            Expr::LitInt(x) => {
                // TODO we assume the int is register-sized
                let x: Result<i64, _> = (*x).try_into();
                if let Ok(n) = x {
                    Some(self.fn_builder.ins().iconst(cty.unwrap(), n))
                } else {
                    panic!("int too wide");
                }
            }
            Expr::LitFloat(x) => Some(match ty {
                Type::Float(TypeFloat::F64) => self.fn_builder.ins().f64const(*x),
                Type::Float(TypeFloat::F32) => self.fn_builder.ins().f32const(*x as f32),
                _ => panic!(),
            }),
            Expr::LitChar(x) => {
                let x = *x as i64;
                Some(self.fn_builder.ins().iconst(cty.unwrap(), x))
            }
            Expr::LitBool(x) => Some(self.fn_builder.ins().bconst(cty.unwrap(), *x)),
            Expr::Block(block) => self.lower_block(block),
            // if-then's without an else
            Expr::If(cond, then_block, None) => {
                let then_cb = self.fn_builder.create_block();
                let final_cb = self.fn_builder.create_block();

                let cond = self.lower_expr(*cond).unwrap();

                self.fn_builder.ins().brnz(cond, then_cb, &[]);
                self.fn_builder.ins().jump(final_cb, &[]);

                self.fn_builder.seal_block(then_cb);
                self.fn_builder.switch_to_block(then_cb);

                self.lower_block(then_block);
                self.fn_builder.ins().jump(final_cb, &[]);

                self.fn_builder.seal_block(final_cb);
                self.fn_builder.switch_to_block(final_cb);

                None
            }
            // if-then-else which can yield values
            Expr::If(cond, then_block, Some(else_expr)) => {
                let then_cb = self.fn_builder.create_block();
                let else_cb = self.fn_builder.create_block();
                let final_cb = self.fn_builder.create_block();

                let cond = self.lower_expr(*cond).unwrap();

                self.fn_builder.ins().brnz(cond, then_cb, &[]);
                self.fn_builder.ins().jump(else_cb, &[]);

                self.fn_builder.seal_block(then_cb);
                self.fn_builder.seal_block(else_cb);

                // then branch
                self.fn_builder.switch_to_block(then_cb);

                if let Some(then_res) = self.lower_block(then_block) {
                    self.fn_builder.ins().jump(final_cb, &[then_res]);
                } else {
                    self.fn_builder.ins().jump(final_cb, &[]);
                }

                // else branch
                self.fn_builder.switch_to_block(else_cb);

                if let Some(else_res) = self.lower_expr(*else_expr) {
                    self.fn_builder.ins().jump(final_cb, &[else_res]);
                } else {
                    self.fn_builder.ins().jump(final_cb, &[]);
                }

                // final, unified block
                self.fn_builder.switch_to_block(final_cb);
                self.fn_builder.seal_block(final_cb);

                let res = if let Some(cty) = cty {
                    self.fn_builder.append_block_param(final_cb, cty);
                    Some(self.fn_builder.block_params(final_cb)[0])
                } else {
                    None
                };

                res
            }
            Expr::While(cond, body_block) => {
                let cond_cb = self.fn_builder.create_block();
                let body_cb = self.fn_builder.create_block();
                let final_cb = self.fn_builder.create_block();

                self.fn_builder.ins().jump(cond_cb, &[]);

                // cond block
                self.fn_builder.switch_to_block(cond_cb);

                let cond = self.lower_expr(*cond).unwrap();

                self.fn_builder.ins().brnz(cond, body_cb, &[]);
                self.fn_builder.ins().jump(final_cb, &[]);

                self.fn_builder.seal_block(body_cb);
                self.fn_builder.seal_block(final_cb);

                // body block
                self.fn_builder.switch_to_block(body_cb);
                self.lower_block(body_block);
                self.fn_builder.ins().jump(cond_cb, &[]);

                self.fn_builder.seal_block(cond_cb);

                // switch to final block
                self.fn_builder.switch_to_block(final_cb);

                None
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
                    if let Some(val) = self.lower_expr(*arg) {
                        c_args.push(val);
                    }
                }

                let call_inst = self.fn_builder.ins().call(fn_ref, &c_args);

                let call_res = self.fn_builder.inst_results(call_inst);
                if call_res.len() > 0 {
                    Some(call_res[0])
                } else {
                    None
                }
            }
            Expr::Call(func, args) => {
                let jit_cb = self.fn_builder.create_block();
                let next_cb = self.fn_builder.create_block();

                let c_sig = lower_sig(func.sig());
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
                    if let Some(val) = self.lower_expr(*arg) {
                        c_args.push(val);
                    }
                }

                let call_inst = self.fn_builder.ins().call_indirect(sig_id, fn_val, &c_args);

                let call_res = self.fn_builder.inst_results(call_inst);
                if call_res.len() > 0 {
                    Some(call_res[0])
                } else {
                    None
                }
            }
            _ => panic!("todo lower expr {:?}", expr),
        }
    }

    fn lower_block(&mut self, block: &Block) -> CVal {
        for expr_id in &block.stmts {
            self.lower_expr(*expr_id);
        }

        if let Some(expr_id) = &block.result {
            self.lower_expr(*expr_id)
        } else {
            None
        }
    }

    fn lower_assign(&mut self, dest_id: u32, src: CVal) -> CVal {
        if let Some(src) = src {
            let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[dest_id as usize];
            let cty = lower_type(*ty);
            match expr {
                Expr::Var(var_id) => {
                    let var = Variable::new(*var_id as usize);
                    self.fn_builder.def_var(var, src);
                }
                _ => panic!("todo lower assignment {:?}", expr),
            }
        }
        src
    }
}
