
use super::func::{Expr, FuncHIR};

#[derive(PartialEq, Debug)]
pub enum VarStorage {
    Register,
    Stack,
    Pointer
}

pub fn get_var_storage(input_fn: &FuncHIR) -> Vec<VarStorage> {
    let var_count = input_fn.vars.len();
    let mut res = Vec::with_capacity(input_fn.exprs.len());
    // TODO non-trivial aggregate arguments use StackPointer
    // TODO non-trivial aggregate vars use StackSlot

    for _ in 0..var_count {
        res.push(VarStorage::Register);
    }

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
