use crate::is_verbose;
use crate::profiler::profile;

use super::infer::FuncTypes;
use super::item::{Function, Item, ItemName, Scope};
use super::types::global::{GlobalType, Signature};
use super::types::common::{TypeKind, IntWidth, IntSign, FloatWidth};

use std::cell::{RefCell, Cell};

pub struct FuncHIR {
    pub root_expr: usize,
    pub exprs: Vec<ExprInfo>,
    pub vars: Vec<u32>, // map into expr list
    pub ret_ty: GlobalType,
    break_index: Vec<(u32, Option<String>)>,
    nested_temporaries: Vec<u32>,
}

#[derive(Debug)]
pub struct ExprInfo {
    pub expr: Expr,
    pub ty: GlobalType
}

impl ExprInfo {
    pub fn new(expr: Expr) -> ExprInfo {
        ExprInfo {
            expr,
            ty: GlobalType::simple(TypeKind::Unknown)
        }
    }
}

fn is_expr_temporary(expr: &Expr) -> bool {
    match expr {
        Expr::Var(..) | Expr::IndexTuple(..) | Expr::DeRef(..) => false,
        Expr::LitFloat(..) | Expr::LitInt(..) | Expr::Block(_) | Expr::Ref(..) | Expr::LitVoid => {
            true
        }
        _ => panic!("temp??? {:?}", expr),
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
            break_index: vec![],
            ret_ty: ty_sig.output.clone(),
            nested_temporaries: vec![],
        };
        profile("lower AST -> HIR", || {
            let mut body = Block::new(Some(parent_scope));
            body.add_args(&mut code, &syn_fn.sig, ty_sig);
            body.add_from_syn(&mut code, &syn_fn.block);

            // todo root? ty_sig.output

            let root = code.push_expr(Expr::Block(Box::new(body)));
            code.root_expr = root as usize;
        });

        if code.break_index.len() > 0 {
            panic!("bad break or continue detected");
        }
        if code.nested_temporaries.len() > 0 {
            panic!("bad temporary detected");
        }

        let (mut types,_) = profile("type infer / setup",||{
            FuncTypes::new(&mut code,ty_sig.inputs.len())
        });

        profile("type infer / solve",||{
            //println!("pre-solve: {}",types.constraint_count());
            for _i in 1..10 {
                let pre_count = types.constraint_count();

                types.solve(&mut code);
    
                let count = types.constraint_count();

                //println!("{} {}",_i,count);

                if count == pre_count {
                    //println!("todo desperate solve! {}",_i);
                    types.solve_desperate(&mut code);
                }

                if count == 0 {
                    break;
                }
            }

            if is_verbose() {
                types.constraint_dump();
            }
    
            types.fix_unknown_primitives();
            types.apply_types(&mut code);
        });

        //profile("type infer", || code.infer_types());
        //panic!("todo type check");

        code
    }

    fn push_expr(&mut self, expr: Expr) -> u32 {
        let id = self.exprs.len() as u32;
        self.exprs.push(ExprInfo::new(expr));
        id
    }

    fn resolve_breaks(&mut self, loop_index: u32, target_label: &Option<syn::Label>) {
        let target_label = target_label.as_ref().map(|t| t.name.ident.to_string());

        self.break_index.retain(|(break_index, source_label)| {
            let mut resolve = false;
            if let Some(source_label) = source_label {
                // has a target
                if let Some(target_label) = &target_label {
                    resolve = source_label == target_label;
                }
            } else {
                // target is the innermost loop (this one)
                resolve = true;
            }

            if resolve {
                let expr = &mut self.exprs[*break_index as usize].expr;
                match expr {
                    Expr::Break(id, _) => *id = loop_index,
                    Expr::Continue(id) => *id = loop_index,
                    _ => panic!("can not resolve a break"),
                }
            }

            !resolve
        });
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
    } else if let syn::Pat::Type(pat_ty) = pat {
        pat_to_name(&pat_ty.pat)
    } else {
        panic!("pattern to name {:?}", pat)
    }
}

fn pat_to_name_and_ty(pat: &syn::Pat, scope: &Scope) -> (String, GlobalType) {
    if let syn::Pat::Type(pat_ty) = pat {
        let ty = GlobalType::from_syn(&pat_ty.ty, scope);
        let name = pat_to_name(&pat_ty.pat);
        (name, ty)
    } else {
        (pat_to_name(pat), GlobalType::simple(TypeKind::Unknown))
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
        if p1 == "_builtin" {
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
            let var_id = code.push_expr(Expr::Var(i as u32, ty.clone()));
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

    fn add_stmt(&mut self, id: u32, code: &mut FuncHIR) {
        if code.nested_temporaries.len() > 0 {
            let tmp_list = std::mem::take(&mut code.nested_temporaries);
            let new_id = code.push_expr(Expr::StmtTmp(id, tmp_list));
            self.stmts.push(new_id);
        } else {
            self.stmts.push(id);
        }
    }

    pub fn add_from_syn(&mut self, code: &mut FuncHIR, syn_block: &syn::Block) {
        //let mut terminate = false;
        let stmt_count = syn_block.stmts.len();
        for (i, stmt) in syn_block.stmts.iter().enumerate() {
            let is_final = i + 1 == stmt_count;
            match stmt {
                syn::Stmt::Expr(syn_expr) => {
                    let expr_id = self.add_expr(code, syn_expr);
                    if is_final {
                        self.result = Some(expr_id);
                    } else {
                        self.add_stmt(expr_id, code);
                    }
                }
                syn::Stmt::Semi(syn_expr, _) => {
                    let expr_id = self.add_expr(code, syn_expr);
                    self.add_stmt(expr_id, code);
                }
                syn::Stmt::Local(syn_local) => {
                    let (name, ty) = pat_to_name_and_ty(&syn_local.pat, &self.scope.borrow());

                    let var_id = code.push_expr(Expr::Var(code.vars.len() as u32, ty));
                    code.vars.push(var_id);

                    self.scope
                        .borrow_mut()
                        .declare(ItemName::Value(name), Item::Local(var_id));

                    self.stmts
                        .push(code.push_expr(Expr::DeclVar(var_id)));

                    if let Some((_, init)) = &syn_local.init {
                        let init_id = self.add_expr(code, &init);
                        let assign_id = code.push_expr(Expr::Assign(var_id, init_id));

                        if code.nested_temporaries.len() > 0 {
                            let tmp_list = std::mem::take(&mut code.nested_temporaries);
                            for tmp in tmp_list {
                                let id = code.push_expr(Expr::DeclTmp(tmp));
                                self.stmts.push(id);
                            }
                        }

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
                code.push_expr(Expr::BinOp(id_l, *op, id_r))
            }
            syn::Expr::Unary(syn::ExprUnary { expr, op, .. }) => {
                let id_arg = self.add_expr(code, expr);
                if let syn::UnOp::Deref(_) = op {
                    code.push_expr(Expr::DeRef(id_arg))
                } else {
                    code.push_expr(Expr::UnOp(id_arg, *op))
                }
            }
            syn::Expr::Reference(syn::ExprReference {
                expr, mutability, ..
            }) => {
                let id_arg = self.add_expr(code, expr);
                let is_mut = mutability.is_some();
                if is_expr_temporary(&code.exprs[id_arg as usize].expr) {
                    code.nested_temporaries.push(id_arg);
                }
                code.push_expr(Expr::Ref(id_arg, is_mut))
            }
            syn::Expr::Field(syn::ExprField { base, member, .. }) => {
                let id_arg = self.add_expr(code, base);
                match member {
                    syn::Member::Named(_name) => panic!("named indexing unsupported"),
                    syn::Member::Unnamed(index) => {
                        code.push_expr(Expr::IndexTuple(id_arg, index.index))
                    }
                }
            }
            syn::Expr::Assign(syn::ExprAssign { left, right, .. }) => {
                let id_l = self.add_expr(code, left);
                let id_r = self.add_expr(code, right);

                code.push_expr(Expr::Assign(id_l, id_r))
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
                let hir_ty = GlobalType::from_syn(ty, &self.scope.borrow());
                let arg = self.add_expr(code, expr);

                code.push_expr(Expr::CastPrimitive(arg, hir_ty))
            }
            syn::Expr::Lit(syn::ExprLit { lit, .. }) => match lit {
                syn::Lit::Int(int) => {
                    let n: u128 = int.base10_parse().unwrap();
                    let suffix = int.suffix();

                    let ty = if suffix.len() != 0 {
                        match suffix {
                            "u8" => Some((IntWidth::Int8,IntSign::Unsigned)),
                            "u16" => Some((IntWidth::Int16,IntSign::Unsigned)),
                            "u32" => Some((IntWidth::Int32,IntSign::Unsigned)),
                            "u64" => Some((IntWidth::Int64,IntSign::Unsigned)),
                            "u128" => Some((IntWidth::Int128,IntSign::Unsigned)),
                            "usize" => Some((IntWidth::IntSize,IntSign::Unsigned)),

                            "i8" => Some((IntWidth::Int8,IntSign::Signed)),
                            "i16" => Some((IntWidth::Int16,IntSign::Signed)),
                            "i32" => Some((IntWidth::Int32,IntSign::Signed)),
                            "i64" => Some((IntWidth::Int64,IntSign::Signed)),
                            "i128" => Some((IntWidth::Int128,IntSign::Signed)),
                            "isize" => Some((IntWidth::IntSize,IntSign::Signed)),

                            _ => panic!("bad suffix {:?}",suffix)
                        }
                    } else {
                        None
                    };
                    code.push_expr(Expr::LitInt(n,ty))
                }
                syn::Lit::Float(float) => {
                    let n: f64 = float.base10_parse().unwrap();
                    let suffix = float.suffix();
                    let ty = if suffix.len() != 0 {
                        match suffix {
                            "f32" => Some(FloatWidth::Float32),
                            "f64" => Some(FloatWidth::Float64),
                            _ => panic!("bad suffix {:?}",suffix)
                        }
                    } else {
                        None
                    };
                    code.push_expr(Expr::LitFloat(n,ty))
                }
                syn::Lit::Bool(syn::LitBool { value, .. }) => {
                    code.push_expr(Expr::LitBool(*value))
                }
                syn::Lit::Char(char) => {
                    let value = char.value();
                    code.push_expr(Expr::LitChar(value))
                }
                _ => panic!("todo handle lit {:?}", lit),
            },
            syn::Expr::Tuple(syn::ExprTuple { elems, .. }) => {
                if elems.len() == 0 {
                    code.push_expr(Expr::LitVoid)
                } else {
                    let fields: Vec<_> =
                        elems.iter().map(|elem| self.add_expr(code, elem)).collect();
                    code.push_expr(Expr::NewTuple(fields))
                }
            }
            // control flow-ish stuff
            syn::Expr::Block(syn::ExprBlock { block, .. })
            | syn::Expr::Unsafe(syn::ExprUnsafe { block, .. }) => {
                let hir_block = self.child_block_from_syn(code, block);
                code.push_expr(Expr::Block(hir_block))
            }
            syn::Expr::If(syn::ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                let id_cond = self.add_expr(code, cond);
                let then_block = self.child_block_from_syn(code, then_branch);
                let id_else = else_branch
                    .as_ref()
                    .map(|(_, else_branch)| self.add_expr(code, else_branch));
                code.push_expr(Expr::If(id_cond, then_block, id_else))
            }
            syn::Expr::While(syn::ExprWhile {
                cond, body, label, ..
            }) => {
                let id_cond = self.add_expr(code, cond);
                let body_block = self.child_block_from_syn(code, body);
                let result = code.push_expr(Expr::While(id_cond, body_block));
                code.resolve_breaks(result, label);
                result
            }
            syn::Expr::Loop(syn::ExprLoop { body, label, .. }) => {
                let body_block = self.child_block_from_syn(code, body);
                let result = code.push_expr(Expr::Loop(body_block));
                code.resolve_breaks(result, label);
                result
            }
            syn::Expr::Break(syn::ExprBreak { label, expr, .. }) => {
                let break_val = expr.as_ref().map(|expr| self.add_expr(code, &expr));
                let label = label.as_ref().map(|l| l.ident.to_string());
                let result = code.push_expr(Expr::Break(std::u32::MAX, break_val));
                code.break_index.push((result, label));
                result
            }
            syn::Expr::Continue(syn::ExprContinue { label, .. }) => {
                let label = label.as_ref().map(|l| l.ident.to_string());
                let result = code.push_expr(Expr::Continue(std::u32::MAX));
                code.break_index.push((result, label));
                result
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
                                code.push_expr(Expr::Call(func, args))
                            } else {
                                panic!("todo call {:?}", item);
                            }
                        } else if let Some(builtin) = path_to_builtin(path) {
                            code.push_expr(Expr::CallBuiltin(builtin, args))
                        } else {
                            panic!("todo call path {:?}", path);
                        }
                    }
                    _ => panic!("attempt to call {:?}", func),
                }
            }
            syn::Expr::Return(syn::ExprReturn { expr, .. }) => {
                if let Some(expr) = expr {
                    let id_arg = self.add_expr(code, expr);
                    code.push_expr(Expr::Return(Some(id_arg)))
                } else {
                    code.push_expr(Expr::Return(None))
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
    Var(u32,GlobalType),
    Block(Box<Block>),
    DeclVar(u32),
    DeclTmp(u32),
    StmtTmp(u32, Vec<u32>),
    BinOp(u32, syn::BinOp, u32),
    BinOpPrimitive(u32, syn::BinOp, u32),
    UnOp(u32, syn::UnOp),
    UnOpPrimitive(u32, syn::UnOp),
    Ref(u32, bool), // 2nd value indicates mutability
    DeRef(u32),
    IndexTuple(u32, u32), // 2nd value indicates index
    LitInt(u128,Option<(IntWidth,IntSign)>),
    LitFloat(f64,Option<FloatWidth>),
    LitBool(bool),
    LitChar(char),
    LitVoid,
    NewTuple(Vec<u32>),
    Assign(u32, u32),
    CastPrimitive(u32,GlobalType),
    If(u32, Box<Block>, Option<u32>),
    While(u32, Box<Block>),
    Loop(Box<Block>),
    Call(&'static Function, Vec<u32>),
    Return(Option<u32>),
    CallBuiltin(String, Vec<u32>),
    Break(u32, Option<u32>),
    Continue(u32),
}
