use std::{collections::HashMap, cell::Cell};

use std::cell::RefCell;
use once_cell::unsync::OnceCell;

use crate::{hir_expr::FuncIR, types::Signature};

#[derive(Debug,Clone,Copy)]
pub enum Item{
    Fn(&'static Function),
    Module(), // not implemented
    Local(u32)
}

impl Item {
    fn from_syn(syn_item: syn::Item, scope: &'static RefCell<Scope>) -> (ItemName,Item) {
        match syn_item {
            syn::Item::Fn(syn_fn) => {
                let base_name = syn_fn.sig.ident.to_string();
                let debug_name = base_name.clone();
                let name = ItemName::Value(base_name);

                let func = Box::new(Function{
                    debug_name,
                    c_fn: Cell::new(std::ptr::null()),

                    syn_fn,
                    parent_scope: scope,
                    sig: OnceCell::new(),
                    ir: OnceCell::new()
                });
                let func = Box::leak(func);

                let item = Item::Fn(func);
                (name,item)
            },
            syn::Item::Mod(syn_mod) => {
                let name = ItemName::Type(syn_mod.ident.to_string());
                let item = Item::Module();
                (name,item)
            },
            _ => panic!("todo handle item => {:?}",syn_item)
        }
    }
}

#[repr(C)]
pub struct Function{
    pub c_fn: Cell<*const u8>,
    pub debug_name: String,
    
    syn_fn: syn::ItemFn,
    parent_scope: &'static RefCell<Scope>,
    sig: OnceCell<Signature>, // todo once-cell?
    ir: OnceCell<FuncIR>,  // todo once-cell?
}

impl std::fmt::Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Function").finish()
    }
}

impl Function {
    // we need:
    // - the type registry
    // - the the scope chain
    pub fn sig(&self) -> &Signature {
        self.sig.get_or_init(|| {
            let scope = self.parent_scope.borrow();
            Signature::from_syn(&self.syn_fn.sig, &scope)
        })
    }

    pub fn ir(&self) -> &FuncIR {
        let sig = self.sig();

        self.ir.get_or_init(|| {
            FuncIR::from_syn(&self.syn_fn, sig, self.parent_scope)
        })
    }
}

#[derive(Eq,PartialEq,Hash,Debug)]
pub enum ItemName {
    Type(String),
    Value(String),
    Macro(String) // contains two sub-namespaces for bang-style macros and attributes
}

pub fn try_path_to_name(path: &syn::Path) -> Option<String> {
    if path.segments.len() == 1 {
        Some(path.segments[0].ident.to_string())
    } else {
        None
    }
}

#[derive(Debug)]
pub struct Scope {
    map: HashMap<ItemName,Item>,
    parent: Option<&'static RefCell<Scope>>
}

impl Scope{
    pub fn new(parent: Option<&'static RefCell<Scope>>) -> &'static RefCell<Scope> {
        let scope = Box::new(RefCell::new(Scope{
            map: HashMap::new(),
            parent
        }));
        let scope: &'static _ = Box::leak(scope);
        scope
    }

    pub fn declare(&mut self, key: ItemName, value: Item) {
        if self.map.insert(key, value).is_some() {
            panic!("duplicate declaration");
        }
    }

    pub fn get(&self, key: &ItemName) -> Option<Item> {
        let res = self.map.get(key).map(|x| *x);
        if res.is_none() {
            if let Some(parent) = self.parent {
                return parent.borrow().get(key);
            }
        }
        res
    }

    pub fn from_syn_file(syn_file: syn::File) -> &'static RefCell<Scope> {
        let scope = Self::new(None);

        for syn_item in syn_file.items {
            let (k,v) = Item::from_syn(syn_item,&scope);
            scope.borrow_mut().declare(k, v);
        }
        scope
    }
}
