// I do not like macros.

pub fn main() {
    write_exec_match();
}

fn write_unary(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{instr}(out, src) => {{
        let x: {ty} = read_stack(stack, src);
        let res = {op};
        write_stack(stack, out, res);
    }}"
    ));
}

fn write_binary(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{instr}(out, lhs, rhs) => {{
        let a: {ty} = read_stack(stack, lhs);
        let b: {ty} = read_stack(stack, rhs);
        let res = {op};
        write_stack(stack, out, res);
    }}"
    ));
}

fn write_shift(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{instr}(out, lhs, rhs) => {{
        let a: {ty} = read_stack(stack, lhs);
        let b: u8 = read_stack(stack, rhs);
        let res = {op};
        write_stack(stack, out, res);
    }}"
    ));
}

fn write_immediate(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{instr}(out, x) => {{
        let res: {ty} = {op};
        write_stack(stack, out, res);
    }}"
    ));
}

fn write_int_ops(signed: &str, unsigned: &str, source: &mut String) {
    let big = signed.to_uppercase();

    if signed == "i128" {
        // I128 constants use up to two instructions, each containing an i64
        source.push_str(&format!(
            "
        Instr::I128_Const(out, x) => {{
            let res: i128 = x as i128;
            write_stack(stack, out, res);
        }}
        Instr::I128_ConstHigh(out, x) => {{
            let res: i64 = x;
            write_stack(stack, out + 8, res);
        }}
        "
        ));
    } else {
        write_immediate(&format!("{}_Const",big), signed, "x", source);
    }
    write_unary(&format!("{}_Mov",big), signed, "x", source);
    write_unary(&format!("{}_Neg",big), signed, "x.wrapping_neg()", source);
    write_unary(&format!("{}_Not",big), signed, "!x", source);
    write_binary(&format!("{}_Eq",big), signed, "a == b", source);
    write_binary(&format!("{}_NotEq",big), signed, "a != b", source);
    write_binary(&format!("{}_Add",big), signed, "a.wrapping_add(b)", source);
    write_binary(&format!("{}_Sub",big), signed, "a.wrapping_sub(b)", source);
    write_binary(&format!("{}_Mul",big), signed, "a.wrapping_mul(b)", source);
    write_binary(&format!("{}_Or",big), signed, "a | b", source);
    write_binary(&format!("{}_And",big), signed, "a & b", source);
    write_binary(&format!("{}_Xor",big), signed, "a ^ b", source);
    write_shift(&format!("{}_ShiftL",big), signed, "a.wrapping_shl(b as _)", source);
    write_binary(&format!("{}_S_Lt",big), signed, "a < b", source);
    write_binary(&format!("{}_S_LtEq",big), signed, "a <= b", source);
    write_binary(&format!("{}_S_Div",big), signed, "a.wrapping_div(b)", source);
    write_binary(&format!("{}_S_Rem",big), signed, "a.wrapping_rem(b)", source);
    write_shift(&format!("{}_S_ShiftR",big), signed, "a.wrapping_shr(b as _)", source);
    write_binary(&format!("{}_U_Lt",big), unsigned, "a < b", source);
    write_binary(&format!("{}_U_LtEq",big), unsigned, "a <= b", source);
    write_binary(&format!("{}_U_Div",big), unsigned, "a.wrapping_div(b)", source);
    write_binary(&format!("{}_U_Rem",big), unsigned, "a.wrapping_rem(b)", source);
    write_shift(&format!("{}_U_ShiftR",big), unsigned, "a.wrapping_shr(b as _)", source);
}

fn write_exec_match() {
    let mut source = String::new();
    source.push_str("match instr {");

    write_int_ops("i8","u8",&mut source);
    write_int_ops("i16","u16",&mut source);
    write_int_ops("i32","u32",&mut source);
    write_int_ops("i64","u64",&mut source);
    write_int_ops("i128","u128",&mut source);

    // widening operations
    source.push_str(
        "
    Instr::I64_S_Widen_32(out, src) => {
        let x: i32 = read_stack(stack, src);
        let res = x as i64;
        write_stack(stack, out, res);
    }
    Instr::I64_U_Widen_32(out, src) => {
        let x: u32 = read_stack(stack, src);
        let res = x as u64;
        write_stack(stack, out, res);
    }
    Instr::I64_S_Widen_16(out, src) => {
        let x: i16 = read_stack(stack, src);
        let res = x as i64;
        write_stack(stack, out, res);
    }
    Instr::I64_U_Widen_16(out, src) => {
        let x: u16 = read_stack(stack, src);
        let res = x as u64;
        write_stack(stack, out, res);
    }
    Instr::I64_S_Widen_8(out, src) => {
        let x: i8 = read_stack(stack, src);
        let res = x as i64;
        write_stack(stack, out, res);
    }
    Instr::I64_U_Widen_8(out, src) => {
        let x: u8 = read_stack(stack, src);
        let res = x as u64;
        write_stack(stack, out, res);
    }
    ",
    );

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
    Instr::BuiltIn_print_int(arg) => {
        let arg: i128 = read_stack(stack, arg);
        crate::builtin::print_int(arg);
    }
    Instr::BuiltIn_print_uint(arg) => {
        let arg: u128 = read_stack(stack, arg);
        crate::builtin::print_uint(arg);
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
