use super::{
    func::{Expr, FuncHIR},
    types::{ComplexType, Type},
};

#[derive(PartialEq, Debug)]
pub enum VarStorage {
    Register,
    Stack,
    Pointer,
    None, // ZST
}

pub fn get_var_storage(input_fn: &FuncHIR) -> Vec<VarStorage> {
    // TODO non-trivial aggregate arguments use StackPointer
    // TODO non-trivial aggregate vars use StackSlot

    let mut res = input_fn
        .vars
        .iter()
        .map(|expr_id| {
            let ty = input_fn.exprs[*expr_id as usize].ty;
            storage_for_type(ty)
        })
        .collect();

    // Referenced vars must be demoted to StackSlot
    for info in &input_fn.exprs {
        if let Expr::Ref(id, _) = info.expr {
            update_storage_for_ref(id, input_fn, &mut res);
        }
    }

    res
}

fn update_storage_for_ref(id: u32, input_fn: &FuncHIR, res: &mut Vec<VarStorage>) {
    let expr = &input_fn.exprs[id as usize].expr;
    if let Expr::Var(var_id) = expr {
        if res[*var_id as usize] == VarStorage::Register {
            res[*var_id as usize] = VarStorage::Stack;
        }
    }
}

// TODO: reuse type lowering code for this instead of
// this duplicate lookup function
fn storage_for_type(ty: Type) -> VarStorage {
    match ty {
        Type::Int(_) => VarStorage::Register,
        Type::Float(_) => VarStorage::Register,
        Type::Char => VarStorage::Register,

        Type::Void => VarStorage::None,

        Type::Complex(ComplexType::Ref(_, _)) => VarStorage::Register,
        _ => panic!("storage for {:?}", ty),
    }
}
