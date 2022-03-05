// based loosly on https://github.com/bytecodealliance/cranelift-jit-demo/blob/main/src/jit.rs

use cranelift::{prelude::*, codegen::Context};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, DataContext, Linkage};
use syn::BinOp;

use crate::{hir_expr::{FuncCode, Block, Expr, ExprInfo}, types::{Signature, Type, TypeInt}};

type CType = cranelift::prelude::Type;
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
    pub fn compile(&mut self, sig: &Signature, code: &FuncCode) -> Result<*const u8, String> {
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

        self.module.define_function(fn_id, &mut self.ctx)
            .map_err(|e| e.to_string())?;
        
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions();
    
        let code = self.module.get_finalized_function(fn_id);
        
        Ok(code)
    }

    fn lower_sig(&mut self, sig: &Signature) {

        for ty in &sig.inputs {
            self.ctx.func.signature.params.push(AbiParam::new(lower_type(*ty)));
        }

        self.ctx.func.signature.returns.push(AbiParam::new(lower_type(sig.output)));
    }
}

fn lower_type(ty: Type) -> CType {
    match ty {
        Type::Int(TypeInt::I32) => types::I32,
        _ => panic!("unknown type")
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
            let ExprInfo{expr,ty} = &self.code.exprs[*index as usize];
            if let Expr::Var(var_id) = expr {
                let var = Variable::new(*var_id as usize);
                let cty = lower_type(*ty);
                self.fn_builder.declare_var(var, cty);
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
        //panic!("stop");
        self.fn_builder.ins().return_(&[res.unwrap()]);

        self.fn_builder.finalize();
    }

    fn lower_expr(&mut self, expr_id: u32) -> CVal {
        let ExprInfo{expr,ty} = &self.code.exprs[expr_id as usize];
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
            Expr::Assign(dest,src) => {
                let src_val = self.lower_expr(*src);
                self.lower_assign(*dest,src_val)
            },
            Expr::BinOpPrimitive(lhs,op,rhs) => {
                let lhs = self.lower_expr(*lhs).unwrap();
                let rhs = self.lower_expr(*rhs).unwrap();

                Some(match *op {
                    BinOp::Add(_) => self.fn_builder.ins().iadd(lhs,rhs),
                    BinOp::Mul(_) => self.fn_builder.ins().imul(lhs,rhs),
                    _ => panic!("todo op {:?}",op)
                })
            },
            Expr::LitInt(x) => {
                // TODO we assume the int is register-sized
                let x: Result<i64,_> = (*x).try_into();
                if let Ok(n) = x {
                    Some( self.fn_builder.ins().iconst(cty,n) )
                } else {
                    panic!("int too wide");
                }
            },
            Expr::Block(block) => {
                for expr_id in &block.stmts {
                    self.lower_expr(*expr_id);
                }

                if let Some(expr_id) = &block.result {
                    self.lower_expr(*expr_id)
                } else {
                    None
                }
            },
            _ => panic!("todo lower expr {:?}",expr)
        }
    }

    fn lower_assign(&mut self, dest_id: u32, src: CVal) -> CVal {
        if let Some(src) = src {
            let ExprInfo{expr,ty} = &self.code.exprs[dest_id as usize];
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
