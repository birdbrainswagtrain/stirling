use super::item::{Function, Item, ItemName, Scope};
use super::types::{Signature, Type};

use std::cell::RefCell;

pub struct FuncHIR {
    pub root_expr: usize,
    pub exprs: Vec<ExprInfo>,
    pub vars: Vec<u32>, // map into expr list
}

pub struct ExprInfo {
    pub expr: Expr,
    pub ty: Type,
    pub is_resolved: bool,
}

impl ExprInfo {
    pub fn new(expr: Expr, ty: Type) -> ExprInfo {
        ExprInfo {
            expr,
            ty,
            is_resolved: false,
        }
    }
}

impl FuncHIR {
    pub fn from_syn(
        syn_fn: &syn::ItemFn,
        ty_sig: &Signature,
        parent_scope: &'static RefCell<Scope>,
    ) -> Self {
        let mut code = FuncHIR {
            root_expr: 0, // invalid, todo fill
            exprs: vec![],
            vars: vec![],
        };
        let mut body = Block::new(Some(parent_scope));
        body.add_args(&mut code, &syn_fn.sig, ty_sig);
        body.add_from_syn(&mut code, &syn_fn.block);

        let root = code.push_expr(Expr::Block(Box::new(body)), ty_sig.output);
        code.root_expr = root as usize;

        code.check();

        code
    }

    fn push_expr(&mut self, expr: Expr, ty: Type) -> u32 {
        let id = self.exprs.len() as u32;
        self.exprs.push(ExprInfo::new(expr, ty));
        id
    }

    pub fn print(&self) {
        println!("ROOT -> {}", self.root_expr);
        for (i, expr) in self.exprs.iter().enumerate() {
            println!("{:4} {:?} :: {:?}", i, expr.expr, expr.ty);
        }
        println!("VARS -> {:?}", self.vars);
    }
}

#[derive(Debug)]
pub struct Block {
    scope: &'static RefCell<Scope>,
    pub stmts: Vec<u32>,
    pub result: Option<u32>,
}

fn pat_to_name(pat: &syn::Pat) -> String {
    if let syn::Pat::Ident(pat_ident) = pat {
        pat_ident.ident.to_string()
    } else {
        panic!("pattern: {:?}", pat);
    }
}

fn path_to_name(path: &syn::Path) -> Option<String> {
    if path.segments.len() == 1 {
        Some(path.segments[0].ident.to_string())
    } else {
        None
    }
}

fn path_to_builtin(path: &syn::Path) -> Option<String> {
    if path.segments.len() == 2 {
        let p1 = path.segments[0].ident.to_string();
        if p1 == "_skitter_builtin" {
            let p2 = path.segments[1].ident.to_string();
            return Some(p2);
        }
    }
    None
}

impl Block {
    pub fn new(parent_scope: Option<&'static RefCell<Scope>>) -> Block {
        Block {
            scope: Scope::new(parent_scope),
            stmts: vec![],
            result: None,
        }
    }

    pub fn add_args(&mut self, code: &mut FuncHIR, syn_sig: &syn::Signature, sig: &Signature) {
        for (i, (syn_arg, ty)) in syn_sig.inputs.iter().zip(&sig.inputs).enumerate() {
            let var_id = code.push_expr(Expr::Var(i as u32), *ty);
            code.vars.push(var_id);

            let name = match syn_arg {
                syn::FnArg::Receiver(_recv) => String::from("self"),
                syn::FnArg::Typed(pt) => pat_to_name(&*pt.pat),
            };
            self.scope
                .borrow_mut()
                .declare(ItemName::Value(name), Item::Local(i as u32));
        }
    }

    pub fn add_from_syn(&mut self, code: &mut FuncHIR, syn_block: &syn::Block) {
        let mut terminate = false;
        for stmt in &syn_block.stmts {
            if terminate {
                panic!("block should have terminated");
            }
            match stmt {
                syn::Stmt::Expr(syn_expr) => {
                    self.result = Some(self.add_expr(code, syn_expr));
                    terminate = true;
                }
                syn::Stmt::Semi(syn_expr, _) => {
                    let expr_id = self.add_expr(code, syn_expr);
                    self.stmts.push(expr_id);
                }
                syn::Stmt::Local(syn_local) => {
                    let ty = Type::Unknown;

                    let var_id = code.push_expr(Expr::Var(code.vars.len() as u32), ty);
                    code.vars.push(var_id);

                    let name = pat_to_name(&syn_local.pat);
                    self.scope
                        .borrow_mut()
                        .declare(ItemName::Value(name), Item::Local(var_id));

                    if let Some((_, init)) = &syn_local.init {
                        let init_id = self.add_expr(code, &init);
                        let assign_id = code.push_expr(Expr::Assign(var_id, init_id), ty);
                        self.stmts.push(assign_id);
                    }
                }
                _ => panic!("todo handle stmt => {:?}", stmt),
            }
        }
    }

    fn add_expr(&mut self, code: &mut FuncHIR, syn_expr: &syn::Expr) -> u32 {
        match syn_expr {
            syn::Expr::Paren(syn::ExprParen { expr, .. }) => self.add_expr(code, expr),
            syn::Expr::Binary(syn::ExprBinary {
                left, op, right, ..
            })
            | syn::Expr::AssignOp(syn::ExprAssignOp {
                left, op, right, ..
            }) => {
                let id_l = self.add_expr(code, left);
                let id_r = self.add_expr(code, right);
                code.push_expr(Expr::BinOp(id_l, *op, id_r), Type::Unknown)
            }
            syn::Expr::Unary(syn::ExprUnary { expr, op, .. }) => {
                let id_arg = self.add_expr(code, expr);
                code.push_expr(Expr::UnOp(id_arg, *op), Type::Unknown)
            }
            syn::Expr::Assign(syn::ExprAssign { left, right, .. }) => {
                let id_l = self.add_expr(code, left);
                let id_r = self.add_expr(code, right);

                code.push_expr(Expr::Assign(id_l, id_r), Type::Unknown)
            }
            syn::Expr::Path(syn::ExprPath { path, .. }) => {
                let name = ItemName::Value(path_to_name(path).expect("unsupported path expr"));

                let scope = self.scope.borrow();
                let item = scope.get(&name);
                if let Some(res) = item {
                    if let Item::Local(id) = res {
                        id
                    } else {
                        panic!("todo path-item {:?}", res);
                    }
                } else {
                    panic!("todo unresolved {:?}", name);
                }
            }
            syn::Expr::Cast(syn::ExprCast { expr, ty, .. }) => {
                let hir_ty = Type::from_syn(ty, &self.scope.borrow());
                let arg = self.add_expr(code, expr);

                code.push_expr(Expr::CastPrimitive(arg), hir_ty)
            }
            syn::Expr::Lit(syn::ExprLit { lit, .. }) => match lit {
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
                }
                syn::Lit::Bool(syn::LitBool { value, .. }) => {
                    code.push_expr(Expr::LitBool(*value), Type::Bool)
                }
                _ => panic!("todo handle lit {:?}", lit),
            },
            // control flow-ish stuff
            syn::Expr::Block(syn::ExprBlock { block, .. }) => {
                let hir_block = self.child_block_from_syn(code, block);
                code.push_expr(Expr::Block(hir_block), Type::Unknown)
            }
            syn::Expr::If(syn::ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                let id_cond = self.add_expr(code, cond);
                let then_block = self.child_block_from_syn(code, then_branch);
                if let Some((_, else_branch)) = else_branch {
                    let id_else = self.add_expr(code, else_branch);
                    code.push_expr(Expr::IfElse(id_cond, then_block, id_else), Type::Unknown)
                } else {
                    // TODO use type unknown so the expr is properly checked
                    panic!("single-side if");
                }
            }
            syn::Expr::While(syn::ExprWhile { cond, body, .. }) => {
                let id_cond = self.add_expr(code, cond);
                let body_block = self.child_block_from_syn(code, body);
                code.push_expr(Expr::While(id_cond, body_block), Type::Void)
            }
            syn::Expr::Call(syn::ExprCall { func, args, .. }) => {
                let args: Vec<_> = args.iter().map(|arg| self.add_expr(code, arg)).collect();

                match func.as_ref() {
                    syn::Expr::Path(syn::ExprPath { path, .. }) => {
                        if let Some(name) = path_to_name(path) {
                            let name = ItemName::Value(name);

                            let scope = self.scope.borrow();
                            let item = scope.get(&name);

                            if let Some(Item::Fn(func)) = item {
                                code.push_expr(Expr::Call(func, args), Type::Unknown)
                            } else {
                                panic!("todo call {:?}", item);
                            }
                        } else if let Some(builtin) = path_to_builtin(path) {
                            code.push_expr(Expr::CallBuiltin(builtin, args), Type::Unknown)
                        } else {
                            panic!("todo call path {:?}", path);
                        }
                    }
                    _ => panic!("attempt to call {:?}", func),
                }
            }
            _ => panic!("todo handle expr => {:?}", syn_expr),
        }
    }

    fn child_block_from_syn(&self, code: &mut FuncHIR, syn_block: &syn::Block) -> Box<Block> {
        let mut block = Block::new(Some(self.scope));
        block.add_from_syn(code, &syn_block);
        Box::new(block)
    }
}

#[derive(Debug)]
pub enum Expr {
    //DeclArg(u32),
    Block(Box<Block>),
    Var(u32),
    BinOp(u32, syn::BinOp, u32),
    BinOpPrimitive(u32, syn::BinOp, u32),
    UnOp(u32, syn::UnOp),
    UnOpPrimitive(u32, syn::UnOp),
    LitInt(u128),
    LitBool(bool),
    Assign(u32, u32),
    CastPrimitive(u32),
    If(u32, Box<Block>),
    IfElse(u32, Box<Block>, u32),
    While(u32, Box<Block>),
    Call(&'static Function, Vec<u32>),
    CallBuiltin(String, Vec<u32>),
}
