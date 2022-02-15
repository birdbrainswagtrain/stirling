use std::collections::HashMap;

use std::rc::{Weak, Rc};
use std::cell::RefCell;
use once_cell::unsync::OnceCell;

use crate::{hir_expr::FuncCode, types::Signature};

#[derive(Debug)]
pub enum Item{
    Fn(Function),
    Local(u32)
}

impl Item {
    fn from_syn(syn_item: syn::Item, scope: &Rc<RefCell<Scope>>) -> (ItemName,Item) {
        match syn_item {
            syn::Item::Fn(syn_fn) => {
                
                let name = ItemName::Value(syn_fn.sig.ident.to_string());
                let item = Item::Fn(Function{
                    syn_fn,
                    parent_scope: Rc::downgrade(scope),
                    sig: OnceCell::new(),
                    code: OnceCell::new()
                });
                (name,item)
            },
            _ => panic!("todo handle item => {:?}",syn_item)
        }
    }
}

pub struct Function{
    syn_fn: syn::ItemFn,
    parent_scope: Weak<RefCell<Scope>>,
    sig: OnceCell<Signature>, // todo once-cell?
    code: OnceCell<FuncCode>  // todo once-cell?
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
            let scope_rc = self.parent_scope.upgrade().unwrap();
            let scope = scope_rc.borrow();
            Signature::from_syn(&self.syn_fn.sig, &scope)
        })
    }

    pub fn code(&mut self) -> &FuncCode {

        /*if self.hir_code.is_none() {
            let mut code = FuncCode::from_syn(&self.syn_fn);
            code.type_check(&self.syn_fn.sig);

            self.hir_code = Some(code);
        }
        
        self.hir_code.as_ref().unwrap()*/
        panic!("stop");
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

#[derive(Debug,Default)]
pub struct Scope {
    map: HashMap<ItemName,Item>,
    parent: Weak<Scope>
}

impl Scope{
    pub fn declare(&mut self, key: ItemName, value: Item) {
        if self.map.insert(key, value).is_some() {
            panic!("duplicate declaration");
        }
    }

    pub fn get(&self, key: &ItemName) -> Option<&Item> {
        self.map.get(key)
    }

    pub fn from_syn_file(syn_file: syn::File) -> Rc<RefCell<Scope>> {
        let mut scope = Rc::new(RefCell::new(Scope{
            map: HashMap::new(),
            parent: Weak::new() // todo should be crate root?
        }));
        for syn_item in syn_file.items {
            let (k,v) = Item::from_syn(syn_item,&scope);
            scope.borrow_mut().declare(k, v);
        }
        scope
    }
}
