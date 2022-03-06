
use crate::hir_items::{Scope, Item, ItemName};
use crate::types::{Type, Signature};

use std::cell::RefCell;
use std::rc::Rc;

pub struct FuncCode{
    pub root_expr: usize,
    pub exprs: Vec<ExprInfo>,
    pub vars: Vec<u32> // map into expr list
}

pub struct ExprInfo{
    pub expr: Expr,
    pub ty: Type
}

impl ExprInfo{
    pub fn new(expr: Expr, ty: Type) -> ExprInfo {
        ExprInfo{expr, ty}
    }
}

impl FuncCode {
    pub fn from_syn(syn_fn: &syn::ItemFn, ty_sig: &Signature, scope: &Scope) -> Self {
        let mut code = FuncCode{
            root_expr: 0, // invalid, todo fill
            exprs: vec!(),
            vars: vec!()
        };
        let mut body: Block = Default::default();
        body.add_args(&mut code, &syn_fn.sig, ty_sig);
        body.add_from_syn(&mut code, &syn_fn.block);

        let root = code.push_expr(Expr::Block(Box::new(body)), Type::Unknown);
        code.root_expr = root as usize;

        code.update_expr_type(root as usize, ty_sig.output);
        code.check();

        code
    }

    fn push_expr(&mut self, expr: Expr, ty: Type) -> u32 {
        let id = self.exprs.len() as u32;
        self.exprs.push(ExprInfo::new(expr,ty));
        id
    }

    pub fn check(&mut self) {

        let mut mutated = true;

        let mut step = 0;

        while mutated {
            mutated = false;

            println!("=================== STEP {} ===================",step);
            step += 1;
            self.print();

            for i in 0..self.exprs.len() {
                let expr_info = &self.exprs[i];
                if expr_info.ty.is_unknown() || expr_info.expr.needs_update() {
                    if self.check_expr(i) {
                        //println!("!!! {} {:?}",i,self.exprs[i].expr);
                        mutated = true;
                    }
                }
            }
        }
    }

    fn check_binary_op_types(&mut self, parent: usize, child1: usize, child2: usize) -> bool {
        let parent_ty = self.exprs[parent].ty;
        let child1_ty = self.exprs[child1].ty;
        let child2_ty = self.exprs[child2].ty;

        if child1_ty != child2_ty || child1_ty != parent_ty {
            if child2_ty.can_upgrade_to(child1_ty) {
                if parent_ty.can_upgrade_to(child1_ty) {
                    self.exprs[parent].ty = child1_ty;
                }
                self.update_expr_type(child2 as usize, child1_ty);
            } else {
                if parent_ty.can_upgrade_to(child2_ty) {
                    self.exprs[parent].ty = child2_ty;
                }
                self.update_expr_type(child1 as usize, child2_ty);
            }
            true
        } else {
            false
        }
    }

    fn check_unary_op_types(&mut self, parent: usize, arg: usize) -> bool {
        let parent_ty = self.exprs[parent].ty;
        let arg_ty = self.exprs[arg].ty;
        if parent_ty.can_upgrade_to(arg_ty) {
            self.exprs[parent].ty = arg_ty;
            true
        } else {
            false
        }
    }

    // returns true if anything was mutated
    fn check_expr(&mut self, index: usize) -> bool {
        let info = &self.exprs[index];

        match info.expr {
            Expr::LitInt(_x) => false,
            Expr::Var(_x) => false,
            Expr::Assign(dst,src) => {
                self.check_binary_op_types(index, dst as usize, src as usize)
            },
            Expr::BinOpPrimitive(lhs,_op,rhs) => {
                // not ALWAYS the right action:
                //  logical ops need bools
                //  bit shifts allow different sized args
                self.check_binary_op_types(index, lhs as usize, rhs as usize)
            },
            Expr::BinOp(lhs,op,rhs) => {
                let lty = self.exprs[lhs as usize].ty;
                let rty = self.exprs[rhs as usize].ty;

                if lty.is_number() && rty.is_number() {
                    self.exprs[index].expr = Expr::BinOpPrimitive(lhs,op,rhs);
                    // not ALWAYS the right action:
                    //  logical ops need bools
                    //  bit shifts allow different sized args
                    self.check_binary_op_types(index, lhs as usize, rhs as usize);
                    // always return true since we switch node types
                    true
                } else {
                    panic!("todo more binary stuff");
                }
            },
            Expr::UnOpPrimitive(arg,_op) => {
                // should always be the right play
                self.check_unary_op_types(index, arg as usize)
            },
            Expr::UnOp(arg,op) => {
                let arg_ty = self.exprs[arg as usize].ty;
                if arg_ty.is_number() {
                    self.exprs[index].expr = Expr::UnOpPrimitive(arg,op);
                    self.check_unary_op_types(index, arg as usize);
                    // always return true since we switch node types
                    true
                } else {
                    panic!("todo more unary stuff");
                }
            },

            Expr::Block(ref block) => {
                if let Some(result_id) = block.result {
                    if self.exprs[index].ty.can_upgrade_to( self.exprs[result_id as usize].ty ) {
                        self.exprs[index].ty = self.exprs[result_id as usize].ty;
                        return true;
                    }
                }
                false
            },
            _ => panic!("todo check {:?}",info.expr)
        }
    }

    fn update_expr_type(&mut self, index: usize, ty: Type) -> bool {
        if self.exprs[index].ty.can_upgrade_to(ty) {
            self.exprs[index].ty = ty;
            
            let info = &self.exprs[index];
            match info.expr {
                Expr::Var(_x) => (),
                Expr::Block(ref block) => {
                    if let Some(res) = block.result {
                        self.update_expr_type(res as usize, ty);
                    } else {
                        panic!("no result!");
                    }
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
    }

    pub fn print(&self) {
        println!("ROOT -> {}",self.root_expr);
        for (i,expr) in self.exprs.iter().enumerate() {
            println!("{:4} {:?} :: {:?}",i,expr.expr,expr.ty);
        }
        println!("VARS -> {:?}",self.vars);
    }
}

#[derive(Debug,Default)]
pub struct Block{
    scope: Rc<RefCell<Scope>>,
    pub stmts: Vec<u32>,
    pub result: Option<u32>
}

fn pat_to_name(pat: &syn::Pat) -> String {
    if let syn::Pat::Ident(pat_ident) = pat {
        pat_ident.ident.to_string()
    } else {
        panic!("pattern: {:?}",pat);
    }
}

impl Block {
    pub fn add_args(&mut self, code: &mut FuncCode, syn_sig: &syn::Signature, sig: &Signature) {
        for (i,(syn_arg,ty)) in syn_sig.inputs.iter().zip(&sig.inputs).enumerate() {

            let var_id = code.push_expr(Expr::Var(i as u32), *ty );
            code.vars.push(var_id);

            let name = match syn_arg {
                syn::FnArg::Receiver(_recv) => String::from("self"),
                syn::FnArg::Typed(pt) => {
                    pat_to_name(&*pt.pat)
                }
            };
            self.scope.borrow_mut().declare(ItemName::Value(name), Item::Local(i as u32));
        }
    }

    pub fn add_from_syn(&mut self, code: &mut FuncCode, syn_block: &syn::Block) {
        let mut terminate = false;
        for stmt in &syn_block.stmts {
            if terminate {
                panic!("block should have terminated");
            }
            match stmt {
                syn::Stmt::Expr(syn_expr) => {
                    self.result = Some( self.add_expr(code,syn_expr) );
                    terminate = true;
                },
                syn::Stmt::Semi(syn_expr,_) => {
                    let expr_id = self.add_expr(code,syn_expr);
                    self.stmts.push(expr_id);
                },
                syn::Stmt::Local(syn_local) => {
                    let ty = Type::Unknown;

                    let var_id = code.push_expr(Expr::Var(code.vars.len() as u32), ty );
                    code.vars.push(var_id);

                    let name = pat_to_name(&syn_local.pat);
                    self.scope.borrow_mut().declare(ItemName::Value(name), Item::Local(var_id));

                    if let Some((_,init)) = &syn_local.init {
                        let init_id = self.add_expr(code, &init);
                        let assign_id = code.push_expr(Expr::Assign(var_id,init_id), ty);
                        self.stmts.push(assign_id);
                    }
                },
                _ => panic!("todo handle stmt => {:?}",stmt)
            }
        }
    }

    fn add_expr(&mut self, code: &mut FuncCode, syn_expr: &syn::Expr) -> u32 {

        match syn_expr {
            syn::Expr::Paren(syn::ExprParen{expr,..}) => self.add_expr(code, expr),
            syn::Expr::Binary(syn::ExprBinary{left,op,right,..}) => {
                let id_l = self.add_expr(code, left);
                let id_r = self.add_expr(code, right);
                code.push_expr(Expr::BinOp(id_l,*op,id_r), Type::Unknown)
            },
            syn::Expr::Unary(syn::ExprUnary{expr,op,..}) => {
                let id_arg = self.add_expr(code, expr);
                code.push_expr(Expr::UnOp(id_arg,*op), Type::Unknown)
            },
            syn::Expr::Assign(syn::ExprAssign{left,right,..}) => {
                let id_l = self.add_expr(code, left);
                let id_r = self.add_expr(code, right);

                code.push_expr(Expr::Assign(id_l,id_r), Type::Unknown)
            },
            syn::Expr::Path(syn::ExprPath{path,..}) => {
                if path.segments.len() == 1 {
                    let name = ItemName::Value(path.segments[0].ident.to_string());
                    let scope = self.scope.borrow();
                    let item = scope.get(&name);
                    if let Some(res) = item {
                        if let Item::Local(id) = res {
                            *id
                        } else {
                            panic!("todo path-item {:?}",res);
                        }
                    } else {
                        panic!("todo unresolved {:?}",name);
                    }
                } else {
                    panic!("todo complex paths");
                }
            },
            syn::Expr::Cast(syn::ExprCast{expr,ty,..}) => {
                let hir_ty = Type::from_syn(ty, &self.scope.borrow());
                let arg = self.add_expr(code, expr);

                code.push_expr(Expr::CastPrimitive(arg), hir_ty)
            },
            syn::Expr::Lit(syn::ExprLit{lit,..}) => {
                match lit {
                    syn::Lit::Int(int) => {
                        let n: u128 = int.base10_parse().unwrap();
                        let suffix = int.suffix();
                        let ty = if suffix.len() != 0 {
                            Type::from_str(suffix).unwrap()
                        } else {
                            Type::IntUnknown
                        };
                        assert!(ty.is_int());
                        code.push_expr(Expr::LitInt(n), ty)
                    },
                    _ => panic!("stop")
                }
            },
            // control flow-ish stuff
            syn::Expr::Block(syn::ExprBlock{block,..}) => {
                let hir_block = Block::from_syn(code, block);
                code.push_expr(Expr::Block(hir_block), Type::Unknown)
            },
            syn::Expr::If(syn::ExprIf{cond,then_branch,else_branch,..}) => {
                let id_cond = self.add_expr(code, cond);
                let then_block = Block::from_syn(code, then_branch);
                if let Some((_,else_branch)) = else_branch {
                    let else_id = self.add_expr(code, else_branch);
                    panic!("dual side if");
                } else {
                    panic!("single-side if");
                }
            },
            _ => panic!("todo handle expr => {:?}",syn_expr)
        }
    }

    fn from_syn(code: &mut FuncCode, syn_block: &syn::Block) -> Box<Block> {
        let mut block: Block = Default::default();
        block.add_from_syn(code, &syn_block);
        Box::new(block)
    }
}

#[derive(Debug)]
pub enum Expr{
    //DeclArg(u32),
    Block(Box<Block>),
    Var(u32),
    BinOp(u32,syn::BinOp,u32),
    BinOpPrimitive(u32,syn::BinOp,u32),
    UnOp(u32,syn::UnOp),
    UnOpPrimitive(u32,syn::UnOp),
    LitInt(u128),
    Assign(u32,u32),
    CastPrimitive(u32),
    If(u32,Box<Block>),
    IfElse(u32,Box<Block>,u32)
}

impl Expr {
    fn needs_update(&self) -> bool {
        match self {
            Expr::BinOp(..) => true,
            Expr::UnOp(..) => true,
            _ => false
        }
    }
}
