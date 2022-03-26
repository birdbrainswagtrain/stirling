// I do not like macros.

pub fn main() {
    write_exec_match();
}

fn write_unary(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!("
    Instr::{instr}(out, src) => {{
        let x: {ty} = read_stack(stack, src);
        let res = {op};
        write_stack(stack, out, res);
    }}"));
}

fn write_binary(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!("
    Instr::{instr}(out, lhs, rhs) => {{
        let a: {ty} = read_stack(stack, lhs);
        let b: {ty} = read_stack(stack, rhs);
        let res = {op};
        write_stack(stack, out, res);
    }}"));
}

fn write_shift(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!("
    Instr::{instr}(out, lhs, rhs) => {{
        let a: {ty} = read_stack(stack, lhs);
        let b: u8 = read_stack(stack, rhs);
        let res = {op};
        write_stack(stack, out, res);
    }}"));
}

fn write_immediate(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!("
    Instr::{instr}(out, x) => {{
        let res: {ty} = {op};
        write_stack(stack, out, res);
    }}"));
}

fn write_exec_match() {
    let mut source = String::new();
    source.push_str("match instr {");

    write_immediate("I32_Const","i32","x",&mut source);
    write_unary("I32_Mov","i32","x",&mut source);
    write_unary("I32_Neg","i32","x.wrapping_neg()",&mut source);
    write_unary("I32_Not","i32","!x",&mut source);

    write_binary("I32_Eq","i32","a == b",&mut source);
    write_binary("I32_NotEq","i32","a != b",&mut source);

    write_binary("I32_Add","i32","a.wrapping_add(b)",&mut source);
    write_binary("I32_Sub","i32","a.wrapping_sub(b)",&mut source);
    write_binary("I32_Mul","i32","a.wrapping_mul(b)",&mut source);

    write_binary("I32_Or","i32","a | b",&mut source);
    write_binary("I32_And","i32","a & b",&mut source);
    write_binary("I32_Xor","i32","a ^ b",&mut source);

    write_shift("I32_ShiftL","i32","a << b",&mut source);

    write_binary("I32_S_Lt","i32","a < b",&mut source);
    write_binary("I32_S_LtEq","i32","a <= b",&mut source);
    write_binary("I32_S_Div","i32","a.wrapping_div(b)",&mut source);
    write_binary("I32_S_Rem","i32","a.wrapping_rem(b)",&mut source);
    write_shift("I32_S_ShiftR","i32","a >> b",&mut source);

    write_binary("I32_U_Lt","u32","a < b",&mut source);
    write_binary("I32_U_LtEq","u32","a <= b",&mut source);
    write_binary("I32_U_Div","u32","a.wrapping_div(b)",&mut source);
    write_binary("I32_U_Rem","u32","a.wrapping_rem(b)",&mut source);
    write_shift("I32_U_ShiftR","u32","a >> b",&mut source);


    source.push_str(
        r#"
    Instr::JumpF(offset, cond) => {
        let x: bool = read_stack(stack, cond);
        if !x {
            pc = (pc as isize + offset as isize) as usize;
            continue;
        }
    }
    Instr::Jump(offset) => {
        pc = (pc as isize + offset as isize) as usize;
        continue;
    }
    Instr::Bad => panic!("encountered bad instruction"),
    Instr::Return => break,
    Instr::BuiltIn_print_i32(arg) => {
        let arg: i32 = read_stack(stack, arg);
        crate::builtin::print_i32(arg);
    }
    Instr::BuiltIn_print_u32(arg) => {
        let arg: u32 = read_stack(stack, arg);
        crate::builtin::print_u32(arg);
    }
    Instr::BuiltIn_print_bool(arg) => {
        let arg: bool = read_stack(stack, arg);
        crate::builtin::print_bool(arg);
    }
    _ => panic!("NYI {:?}",instr)
}"#,
    );

    std::fs::write("src/vm/_exec_match.txt", source).unwrap();
}
