use crate::hir_items::{Scope, Item, ItemName};
use crate::types::Type;

use std::cell::RefCell;
use std::rc::Rc;


pub struct FuncCode{
    body: Block,
    exprs: Vec<Expr>,
    types: Option<Vec<Type>>
}

impl FuncCode {
    pub fn from_syn(syn_fn: &syn::ItemFn) -> Self {
        let mut code = FuncCode{
            body: Default::default(),
            exprs: vec!(),
            types: None
        };
        code.body.add_args(&mut code.exprs, &syn_fn.sig);
        code.body.add_from_syn(&mut code.exprs, &syn_fn.block);
        code
    }

    pub fn type_check(&mut self, syn_sig: &syn::Signature) {
        if self.types.is_none() {
            let types = vec!(Type::Unknown;self.exprs.len());

            let mut mutated = true;

            while mutated {
                mutated = false;

                for i in 0..types.len() {
                    let ty = types[i];
                    if ty.is_unknown() {
                        let new_ty = Type::resolve(&self.exprs[i], &types, ty);
                        if ty != new_ty {
                            mutated = true;
                        }
                    }
                }

                // todo update child types
            }
        }
    }

    pub fn print(&self) {
        for (i,expr) in self.exprs.iter().enumerate() {
            println!("{:4} {:?}",i,expr);
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

impl Block {
    pub fn add_args(&mut self, exprs: &mut Vec<Expr>, sig: &syn::Signature) {
        for (i,arg) in sig.inputs.iter().enumerate() {
            exprs.push(Expr::DeclArg(i as u32));
            let name = match arg {
                syn::FnArg::Receiver(x) => String::from("self"),
                syn::FnArg::Typed(x) => {
                    if let syn::Pat::Ident(pat_ident) = &*x.pat {
                        pat_ident.ident.to_string()
                    } else {
                        panic!("pattern: {:?}",x.pat);
                    }
                }
            };
            self.scope.borrow_mut().declare(ItemName::Value(name), Item::Local(i as u32));
        }
    }

    pub fn add_from_syn(&mut self, exprs: &mut Vec<Expr>, syn_block: &syn::Block) {
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
                _ => panic!("todo handle stmt => {:?}",stmt)
            }
        }
    }

    fn add_expr(&mut self, exprs: &mut Vec<Expr>, syn_expr: &syn::Expr) -> u32 {
        fn push_expr(exprs: &mut Vec<Expr>, expr: Expr) -> u32 {
            let id = exprs.len() as u32;
            exprs.push(expr);
            id
        }

        match syn_expr {
            syn::Expr::Paren(syn::ExprParen{expr,..}) => self.add_expr(exprs, expr),
            syn::Expr::Binary(syn::ExprBinary{left,op,right,..}) => {
                let id_l = self.add_expr(exprs, left);
                let id_r = self.add_expr(exprs, right);
                push_expr(exprs, Expr::BinOp(id_l,*op,id_r))
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
                        push_expr(exprs, Expr::LitInt(n))
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
    DeclVar(u32),
    BinOp(u32,syn::BinOp,u32),
    LitInt(u128)
}
