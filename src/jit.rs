// based loosly on https://github.com/bytecodealliance/cranelift-jit-demo/blob/main/src/jit.rs

use cranelift::{prelude::*, codegen::Context};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, DataContext, Linkage};
use syn::{BinOp, UnOp};

use crate::{hir_expr::{FuncCode, Block, Expr, ExprInfo}, types::{Signature, Type, TypeInt}};

type CType = Option<cranelift::prelude::Type>;
type CVal = Option<Value>;

pub struct JIT {
    module: JITModule,
    ctx: Context,
    builder_ctx: FunctionBuilderContext,
    data_ctx: DataContext
}

impl Default for JIT {
    fn default() -> Self {
        let jit_builder = JITBuilder::new(cranelift_module::default_libcall_names());
        let module = JITModule::new(jit_builder);

        let ctx = module.make_context();
        let builder_ctx = FunctionBuilderContext::new();
        let data_ctx = DataContext::new();

        Self{
            module,
            ctx,
            builder_ctx,
            data_ctx
        }
    }
}

impl JIT {
    pub fn compile(&mut self, sig: &Signature, code: &FuncCode) -> Result<(*const u8,usize), String> {
        let fn_id = self.module.declare_function("butt", Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| e.to_string())?;

        {
            self.lower_sig(sig);

            //let mut fn_builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let mut jit_func = JITFunc{
                code,
                fn_builder: FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx)
            };

            jit_func.compile();
        }

        let compiled_fn = self.module.define_function(fn_id, &mut self.ctx)
            .map_err(|e| e.to_string())?;
        let size = compiled_fn.size as usize;
        
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions();
    
        let code = self.module.get_finalized_function(fn_id);
        
        Ok((code,size))
    }

    fn lower_sig(&mut self, sig: &Signature) {

        for ty in &sig.inputs {
            if let Some(cty) = lower_type(*ty) {
                self.ctx.func.signature.params.push(AbiParam::new(cty));
            }
        }

        if let Some(cty) = lower_type(sig.output) {
            self.ctx.func.signature.returns.push(AbiParam::new(cty));
        }
    }
}

fn lower_type(ty: Type) -> CType {
    match ty {
        Type::Int(TypeInt::I32) | Type::Int(TypeInt::U32) => Some(types::I32),
        Type::Int(TypeInt::I16) | Type::Int(TypeInt::U16) => Some(types::I16),
        Type::Bool => Some(types::B1),
        Type::Void => None,
        _ => panic!("unknown type")
    }
}

#[derive(Debug,Copy,Clone)]
enum LowBinOp {
    Add, Sub, Mul, Div, Rem,
    //BitAnd, BitOr, BitXor
    Gt, Lt
}

impl LowBinOp {
    fn int_cond_code(&self, sign: bool) -> IntCC {
        match (self,sign) {
            (LowBinOp::Gt,true) => IntCC::SignedGreaterThan,
            (LowBinOp::Lt,true) => IntCC::SignedLessThan,
            _ => panic!("cond code for {:?} {}",self,sign)
        }
    }
}

fn lower_bin_op(op: &BinOp) -> (LowBinOp,bool) {
    match op {
        BinOp::Add(_) =>    (LowBinOp::Add, false),
        BinOp::Sub(_) =>    (LowBinOp::Sub, false),
        BinOp::Mul(_) =>    (LowBinOp::Mul, false),
        BinOp::Div(_) =>    (LowBinOp::Div, false),
        BinOp::Rem(_) =>    (LowBinOp::Rem, false),

        BinOp::AddEq(_) =>  (LowBinOp::Add, true),
        BinOp::SubEq(_) =>  (LowBinOp::Sub, true),
        BinOp::MulEq(_) =>  (LowBinOp::Mul, true),
        BinOp::DivEq(_) =>  (LowBinOp::Div, true),
        BinOp::RemEq(_) =>  (LowBinOp::Rem, true),

        BinOp::Gt(_) => (LowBinOp::Gt, false),
        BinOp::Lt(_) => (LowBinOp::Lt, false),

        _ => panic!("can't lower op {:?}",op)
    }
}

struct JITFunc<'a> {
    code: &'a FuncCode,
    fn_builder: FunctionBuilder<'a>,
}

impl<'a> JITFunc<'a> {
    pub fn compile(&mut self) {
        let entry_block = self.fn_builder.create_block();

        for index in self.code.vars.iter() {
            let ExprInfo{expr,ty,..} = &self.code.exprs[*index as usize];
            if let Expr::Var(var_id) = expr {
                let cty = lower_type(*ty);
                if let Some(cty) = lower_type(*ty) {
                    let var = Variable::new(*var_id as usize);
                    self.fn_builder.declare_var(var, cty);
                }
            } else {
                panic!("can not create var from {:?}",expr);
            }
        }

        self.fn_builder.append_block_params_for_function_params(entry_block);
        self.fn_builder.switch_to_block(entry_block);
        self.fn_builder.seal_block(entry_block);

        let param_vals = Vec::from(self.fn_builder.block_params(entry_block));
        for (var_id,val) in param_vals.iter().enumerate() {
            let var = Variable::new(var_id);
            self.fn_builder.def_var(var, *val);
        }
        
        let res = self.lower_expr(self.code.root_expr as u32);
        self.fn_builder.ins().return_(&[res.unwrap()]);

        self.fn_builder.finalize();
    }

    fn lower_expr(&mut self, expr_id: u32) -> CVal {
        let ExprInfo{expr,ty,..} = &self.code.exprs[expr_id as usize];
        let cty = lower_type(*ty);
        match expr {
            /*Expr::DeclVar(n) => {
                let ty = lower_type(expr.ty);
                let var = Variable::new(n);
                self.fn_builder.declare_var(var, ty);
            },*/
            Expr::Var(var_id) => {
                let var = Variable::new(*var_id as usize);
                Some(self.fn_builder.use_var(var))
            },
            Expr::CastPrimitive(arg) => {
                let src_ty = self.code.exprs[*arg as usize].ty;

                let arg = self.lower_expr(*arg).unwrap();

                if src_ty.is_int() && ty.is_int() {
                    let size_src = src_ty.byte_size();
                    let size_dest = ty.byte_size();

                    if size_src == size_dest {
                        Some(arg)
                    } else if size_src < size_dest {
                        // widening: our type of extension is determined by the source type
                        if src_ty.is_signed() {
                            Some( self.fn_builder.ins().sextend(cty.unwrap(),arg) )
                        } else {
                            Some( self.fn_builder.ins().uextend(cty.unwrap(),arg) )
                        }
                    } else {
                        // narrowing
                        Some( self.fn_builder.ins().ireduce(cty.unwrap(),arg) )
                    }
                } else {
                    panic!("non integer casts nyi");
                }
            },
            Expr::Assign(dest,src) => {
                let src_val = self.lower_expr(*src);
                self.lower_assign(*dest,src_val)
            },
            Expr::BinOpPrimitive(lhs,op,rhs) => {
                let lval = self.lower_expr(*lhs).unwrap();
                let rval = self.lower_expr(*rhs).unwrap();

                let (op,is_assign) = lower_bin_op(op);

                let res_val = Some(match (ty,op) {
                    (Type::Int(_),LowBinOp::Add) => self.fn_builder.ins().iadd(lval,rval),
                    (Type::Int(_),LowBinOp::Sub) => self.fn_builder.ins().isub(lval,rval),
                    (Type::Int(_),LowBinOp::Mul) => self.fn_builder.ins().imul(lval,rval),
                    (Type::Int(_),LowBinOp::Div) => 
                        if ty.is_signed() {
                            self.fn_builder.ins().sdiv(lval,rval)
                        } else {
                            self.fn_builder.ins().udiv(lval,rval)
                        },
                    (Type::Int(_),LowBinOp::Rem) => 
                        if ty.is_signed() {
                            self.fn_builder.ins().srem(lval,rval)
                        } else {
                            self.fn_builder.ins().urem(lval,rval)
                        },
                    (_,LowBinOp::Gt) | (_,LowBinOp::Lt) => {
                        let arg_ty = self.code.exprs[*lhs as usize].ty;
                        match arg_ty {
                            Type::Int(_) => {
                                self.fn_builder.ins().icmp(op.int_cond_code(arg_ty.is_signed()), lval,rval)
                            },
                            _ => panic!("can't compare {:?}",arg_ty)
                        }
                    }
                    
                    _ => panic!("can't lower primitive op {:?} {:?}",op,ty)
                });

                if is_assign {
                    self.lower_assign(*lhs,res_val );
                }

                res_val
            },
            Expr::UnOpPrimitive(arg,op) => {
                let arg = self.lower_expr(*arg).unwrap();

                Some(match *op {
                    UnOp::Neg(_) => self.fn_builder.ins().ineg(arg),
                    _ => panic!("todo op {:?}",op)
                })
            },
            Expr::LitInt(x) => {
                // TODO we assume the int is register-sized
                let x: Result<i64,_> = (*x).try_into();
                if let Ok(n) = x {
                    Some( self.fn_builder.ins().iconst(cty.unwrap(),n) )
                } else {
                    panic!("int too wide");
                }
            },
            Expr::LitBool(x) => {
                Some( self.fn_builder.ins().bconst(cty.unwrap(),*x) )
            },
            Expr::Block(block) => {
                self.lower_block(block)
            },
            Expr::IfElse(cond,then_block,else_expr) => {
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
                    Some( self.fn_builder.block_params(final_cb)[0] )
                } else {
                    None
                };

                res
            },
            Expr::While(cond,body_block) => {
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
            },
            _ => panic!("todo lower expr {:?}",expr)
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
            let ExprInfo{expr,ty,..} = &self.code.exprs[dest_id as usize];
            let cty = lower_type(*ty);
            match expr {
                Expr::Var(var_id) => {
                    let var = Variable::new(*var_id as usize);
                    self.fn_builder.def_var(var, src);
                },
                _ => panic!("todo lower assignment {:?}",expr)
            }
        }
        src
    }
}
