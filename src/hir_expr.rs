
use crate::hir_items::{Scope, Item, ItemName};
use crate::types::{Type, Signature};

use std::cell::RefCell;
use std::rc::Rc;

pub struct FuncCode{
    body: Block,
    exprs: Vec<ExprInfo>
}

pub struct ExprInfo{
    expr: Expr,
    ty: Type,
    op_info: OpInfo
}

impl ExprInfo{
    pub fn new(expr: Expr, ty: Type) -> ExprInfo {
        ExprInfo{expr, ty, op_info: OpInfo::None}
    }
}

#[derive(Debug,PartialEq)]
enum OpInfo{
    None,
    //PrimitiveOp
}

impl FuncCode {
    pub fn from_syn(syn_fn: &syn::ItemFn, sig: &Signature, scope: &Scope) -> Self {
        let mut code = FuncCode{
            body: Default::default(),
            exprs: vec!()
        };
        code.body.add_args(&mut code.exprs, &syn_fn.sig, sig);
        code.body.add_from_syn(&mut code.exprs, &syn_fn.block);

        code.check(sig);

        code
    }

    pub fn check(&mut self, sig: &Signature) {

        let mut mutated = true;

        while mutated {
            println!("pass...");
            mutated = false;

            for i in 0..self.exprs.len() {
                let expr_info = &self.exprs[i];
                if expr_info.ty.is_unknown() {
                    if self.check_expr(i) {
                        //println!("!!! {} {:?}",i,self.exprs[i].expr);
                        mutated = true;
                    }
                }
            }

            // TODO: swap in a return instead of having a trailing expr?
            if let Some(res) = self.body.result {
                if self.update_expr_type(res as usize, sig.output) {
                    mutated = true;
                }
            }
        }
    }

    fn update_binary_op_types(&mut self, parent: usize, child1: usize, child2: usize) -> bool {
        let parent_ty = self.exprs[parent].ty;
        let child1_ty = self.exprs[child1].ty;
        let child2_ty = self.exprs[child2].ty;

        if child1_ty != child2_ty || child1_ty != parent_ty {
            println!("update {:?} -> {:?} {:?} {:?}",self.exprs[parent].expr,child1_ty,child2_ty,parent_ty);
            if child1_ty.more_specific_than(child2_ty) {
                println!("path1");
                self.exprs[parent].ty = child1_ty;
                self.update_expr_type(child2 as usize, child1_ty);
            } else {
                println!("path2");
                self.exprs[parent].ty = child2_ty;
                self.update_expr_type(child1 as usize, child2_ty);
            }
            true
        } else {
            false
        }
    }

    // returns true if anything was mutated
    fn check_expr(&mut self, index: usize) -> bool {
        let info = &self.exprs[index];
        let current_ty = info.ty;
        match info.expr {
            Expr::LitInt(_x) => false,
            Expr::DeclVar() => false,
            Expr::Assign(dst,src) => {
                self.update_binary_op_types(index, dst as usize, src as usize)
            },
            Expr::BinOpPrimitive(lhs,op,rhs) => {
                // not ALWAYS the right action:
                //  logical ops need bools
                //  bit shifts allow different sized args
                self.update_binary_op_types(index, lhs as usize, rhs as usize)
            },
            Expr::BinOp(lhs,op,rhs) => {
                let lty = self.exprs[lhs as usize].ty;
                let rty = self.exprs[rhs as usize].ty;

                if lty.is_numeric_primitive() && rty.is_numeric_primitive() {
                    self.exprs[index].expr = Expr::BinOpPrimitive(lhs,op,rhs);
                    // not ALWAYS the right action:
                    //  logical ops need bools
                    //  bit shifts allow different sized args
                    self.update_binary_op_types(index, lhs as usize, rhs as usize);
                    // always return true since we switch node types
                    true
                } else {
                    panic!("todo trait lookup");
                }
            },
            _ => panic!("todo check {:?}",info.expr)
        }
    }

    fn update_expr_type(&mut self, index: usize, ty: Type) -> bool {
        if ty.more_specific_than( self.exprs[index].ty ) {
            self.exprs[index].ty = ty;
            
            let info = &self.exprs[index];
            match info.expr {
                Expr::DeclVar() => (),
                Expr::LitInt(_x) => {
                    assert!(ty.is_numeric_primitive());
                },
                Expr::BinOpPrimitive(lhs,_op,rhs) => {
                    // todo this is NOT correct for all operators
                    self.update_expr_type(lhs as usize, ty);
                    self.update_expr_type(rhs as usize, ty);
                },
                _ => panic!("todo update {:?}",info.expr)
            }
            return true;
        }
        false
    }

    pub fn print(&self) {
        for (i,expr) in self.exprs.iter().enumerate() {
            println!("{:4} {:?} :: {:?} ~~ {:?}",i,expr.expr,expr.ty,expr.op_info);
        }
        println!("=> {:?}",self.body);
    }
}

#[derive(Debug,Default)]
pub struct Block{
    scope: Rc<RefCell<Scope>>,
    stmts: Vec<u32>,
    result: Option<u32>
}

fn push_expr(exprs: &mut Vec<ExprInfo>, expr: Expr, ty: Type) -> u32 {
    let id = exprs.len() as u32;
    exprs.push(ExprInfo::new(expr,ty));
    id
}

fn pat_to_name(pat: &syn::Pat) -> String {
    if let syn::Pat::Ident(pat_ident) = pat {
        pat_ident.ident.to_string()
    } else {
        panic!("pattern: {:?}",pat);
    }
}

impl Block {
    pub fn add_args(&mut self, exprs: &mut Vec<ExprInfo>, syn_sig: &syn::Signature, sig: &Signature) {
        for (i,(syn_arg,ty)) in syn_sig.inputs.iter().zip(&sig.inputs).enumerate() {
            exprs.push( ExprInfo::new(Expr::DeclArg(i as u32),*ty) );
            let name = match syn_arg {
                syn::FnArg::Receiver(_recv) => String::from("self"),
                syn::FnArg::Typed(pt) => {
                    pat_to_name(&*pt.pat)
                }
            };
            self.scope.borrow_mut().declare(ItemName::Value(name), Item::Local(i as u32));
        }
    }

    pub fn add_from_syn(&mut self, exprs: &mut Vec<ExprInfo>, syn_block: &syn::Block) {
        let mut terminate = false;
        for stmt in &syn_block.stmts {
            if terminate {
                panic!("block should have terminated");
            }
            match stmt {
                syn::Stmt::Expr(syn_expr) => {
                    self.result = Some( self.add_expr(exprs,syn_expr) );
                    terminate = true;
                },
                syn::Stmt::Local(syn_local) => {
                    let ty = Type::Unknown;

                    let var_id = push_expr(exprs, Expr::DeclVar(), ty);
                    let name = pat_to_name(&syn_local.pat);
                    self.scope.borrow_mut().declare(ItemName::Value(name), Item::Local(var_id));
                    // the decl itself is pushed to the stmts list
                    self.stmts.push(var_id);

                    if let Some((_,init)) = &syn_local.init {
                        let init_id = self.add_expr(exprs, &init);
                        let assign_id = push_expr(exprs, Expr::Assign(var_id,init_id), ty);
                        self.stmts.push(assign_id);
                    }
                },
                _ => panic!("todo handle stmt => {:?}",stmt)
            }
        }
    }

    fn add_expr(&mut self, exprs: &mut Vec<ExprInfo>, syn_expr: &syn::Expr) -> u32 {

        match syn_expr {
            syn::Expr::Paren(syn::ExprParen{expr,..}) => self.add_expr(exprs, expr),
            syn::Expr::Binary(syn::ExprBinary{left,op,right,..}) => {
                let id_l = self.add_expr(exprs, left);
                let id_r = self.add_expr(exprs, right);
                push_expr(exprs, Expr::BinOp(id_l,*op,id_r), Type::Unknown)
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
            syn::Expr::Lit(syn::ExprLit{lit,..}) => {
                match lit {
                    syn::Lit::Int(int) => {
                        let n: u128 = int.base10_parse().unwrap();
                        let suffix = int.suffix();
                        if suffix.len() != 0 {
                            panic!("todo lit suffix");
                        }
                        push_expr(exprs, Expr::LitInt(n), Type::IntUnknown)
                    },
                    _ => panic!("stop")
                }
            }
            _ => panic!("todo handle expr => {:?}",syn_expr)
        }
    }

    pub fn from_syn(exprs: &mut Vec<Expr>, syn_block: &syn::Block) -> Block {
        panic!("fixme")
    }
}

#[derive(Debug)]
pub enum Expr{
    DeclArg(u32),
    DeclVar(),
    BinOp(u32,syn::BinOp,u32),
    BinOpPrimitive(u32,syn::BinOp,u32),
    LitInt(u128),
    Assign(u32,u32),
}
