// I do not like macros.

pub fn main() {
    write_exec_match();
}

fn write_unary(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{instr}(out, src) => {{
        let x: {ty} = read_stack(stack, *src);
        let res = {op};
        write_stack(stack, *out, res);
    }}"
    ));
}

fn write_binary(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{instr}(out, lhs, rhs) => {{
        let a: {ty} = read_stack(stack, *lhs);
        let b: {ty} = read_stack(stack, *rhs);
        let res = {op};
        write_stack(stack, *out, res);
    }}"
    ));
}

fn write_shift(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{instr}(out, lhs, rhs) => {{
        let a: {ty} = read_stack(stack, *lhs);
        let b: u8 = read_stack(stack, *rhs);
        let res = {op};
        write_stack(stack, *out, res);
    }}"
    ));
}

fn write_immediate(instr: &str, ty: &str, op: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{instr}(out, x) => {{
        let x = *x;
        let res: {ty} = {op};
        write_stack(stack, *out, res);
    }}"
    ));
}

fn write_widen(dst_bits: i32, src_bits: i32, signed: bool, source: &mut String) {
    let sign_char = if signed { 'S' } else { 'U' };
    let ty_char = if signed { 'i' } else { 'u' };
    source.push_str(&format!(
        "
    Instr::I{dst_bits}_{sign_char}_Widen_{src_bits}(out, src) => {{
        let x: {ty_char}{src_bits} = read_stack(stack, *src);
        let res = x as {ty_char}{dst_bits};
        write_stack(stack, *out, res);
    }}"));
}

fn write_cast(name: &str, dst_ty: &str, src_ty: &str, source: &mut String) {
    source.push_str(&format!(
        "
    Instr::{name}(out, src) => {{
        let x: {src_ty} = read_stack(stack, *src);
        let res = x as {dst_ty};
        write_stack(stack, *out, res);
    }}"));
}

fn write_int_ops(signed: &str, unsigned: &str, source: &mut String) {
    let big = signed.to_uppercase();

    if signed == "i128" {
        // I128 constants contain a pointer
        write_immediate(&format!("{}_Const",big), signed, "*x", source);
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

fn write_float_ops(ty: &str, source: &mut String) {
    let big = ty.to_uppercase();
    write_immediate(&format!("{}_Const",big), ty, "x", source);
    write_unary(&format!("{}_Neg",big), ty, "-x", source);
    write_binary(&format!("{}_Add",big), ty, "a + b", source);
    write_binary(&format!("{}_Sub",big), ty, "a - b", source);
    write_binary(&format!("{}_Mul",big), ty, "a * b", source);
    write_binary(&format!("{}_Div",big), ty, "a / b", source);
    write_binary(&format!("{}_Rem",big), ty, "a % b", source);

    write_binary(&format!("{}_Eq",big), ty, "a == b", source);
    write_binary(&format!("{}_NotEq",big), ty, "a != b", source);

    write_binary(&format!("{}_Lt",big), ty, "a < b", source);
    write_binary(&format!("{}_LtEq",big), ty, "a <= b", source);
    write_binary(&format!("{}_Gt",big), ty, "a > b", source);
    write_binary(&format!("{}_GtEq",big), ty, "a >= b", source);
}

fn write_exec_match() {
    let mut source = String::new();
    source.push_str("match instr {");

    write_int_ops("i8","u8",&mut source);
    write_int_ops("i16","u16",&mut source);
    write_int_ops("i32","u32",&mut source);
    write_int_ops("i64","u64",&mut source);
    write_int_ops("i128","u128",&mut source);

    write_float_ops("f64",&mut source);
    write_float_ops("f32",&mut source);

    // Integer bitwise not won't work for bools
    write_unary("Bool_Not", "bool", "!x", &mut source);

    // float casts
    write_cast("F64_From_F32","f64","f32",&mut source);
    write_cast("F64_From_I8_S","f64","i8",&mut source);
    write_cast("F64_From_I8_U","f64","u8",&mut source);
    write_cast("F64_From_I16_S","f64","i16",&mut source);
    write_cast("F64_From_I16_U","f64","u16",&mut source);
    write_cast("F64_From_I32_S","f64","i32",&mut source);
    write_cast("F64_From_I32_U","f64","u32",&mut source);
    write_cast("F64_From_I64_S","f64","i64",&mut source);
    write_cast("F64_From_I64_U","f64","u64",&mut source);
    write_cast("F64_From_I128_S","f64","i128",&mut source);
    write_cast("F64_From_I128_U","f64","u128",&mut source);

    write_cast("F64_Into_I8_S","i8","f64",&mut source);
    write_cast("F64_Into_I8_U","u8","f64",&mut source);
    write_cast("F64_Into_I16_S","i16","f64",&mut source);
    write_cast("F64_Into_I16_U","u16","f64",&mut source);
    write_cast("F64_Into_I32_S","i32","f64",&mut source);
    write_cast("F64_Into_I32_U","u32","f64",&mut source);
    write_cast("F64_Into_I64_S","i64","f64",&mut source);
    write_cast("F64_Into_I64_U","u64","f64",&mut source);
    write_cast("F64_Into_I128_S","i128","f64",&mut source);
    write_cast("F64_Into_I128_U","u128","f64",&mut source);
    
    write_cast("F32_From_F64","f32","f64",&mut source);
    write_cast("F32_From_I8_S","f32","i8",&mut source);
    write_cast("F32_From_I8_U","f32","u8",&mut source);
    write_cast("F32_From_I16_S","f32","i16",&mut source);
    write_cast("F32_From_I16_U","f32","u16",&mut source);
    write_cast("F32_From_I32_S","f32","i32",&mut source);
    write_cast("F32_From_I32_U","f32","u32",&mut source);
    write_cast("F32_From_I64_S","f32","i64",&mut source);
    write_cast("F32_From_I64_U","f32","u64",&mut source);
    write_cast("F32_From_I128_S","f32","i128",&mut source);
    write_cast("F32_From_I128_U","f32","u128",&mut source);

    write_cast("F32_Into_I8_S","i8","f32",&mut source);
    write_cast("F32_Into_I8_U","u8","f32",&mut source);
    write_cast("F32_Into_I16_S","i16","f32",&mut source);
    write_cast("F32_Into_I16_U","u16","f32",&mut source);
    write_cast("F32_Into_I32_S","i32","f32",&mut source);
    write_cast("F32_Into_I32_U","u32","f32",&mut source);
    write_cast("F32_Into_I64_S","i64","f32",&mut source);
    write_cast("F32_Into_I64_U","u64","f32",&mut source);
    write_cast("F32_Into_I128_S","i128","f32",&mut source);
    write_cast("F32_Into_I128_U","u128","f32",&mut source);

    // widening operations
    write_widen(16,8,true,&mut source);
    write_widen(16,8,false,&mut source);

    write_widen(32,16,true,&mut source);
    write_widen(32,16,false,&mut source);
    write_widen(32,8,true,&mut source);
    write_widen(32,8,false,&mut source);

    write_widen(64,32,true,&mut source);
    write_widen(64,32,false,&mut source);
    write_widen(64,16,true,&mut source);
    write_widen(64,16,false,&mut source);
    write_widen(64,8,true,&mut source);
    write_widen(64,8,false,&mut source);

    write_widen(128,64,true,&mut source);
    write_widen(128,64,false,&mut source);
    write_widen(128,32,true,&mut source);
    write_widen(128,32,false,&mut source);
    write_widen(128,16,true,&mut source);
    write_widen(128,16,false,&mut source);
    write_widen(128,8,true,&mut source);
    write_widen(128,8,false,&mut source);

    source.push_str(
        r#"
    Instr::JumpF(offset, cond) => {
        let x: bool = read_stack(stack, *cond);
        if !x {
            pc = (pc as isize + *offset as isize) as usize;
            continue;
        }
    }
    Instr::Jump(offset) => {
        pc = (pc as isize + *offset as isize) as usize;
        continue;
    }
    Instr::Bad => panic!("encountered bad instruction"),
    Instr::Return => break,
    Instr::Call(base,func) => {
        crate::vm::exec(func,stack.offset(*base as isize));
        //panic!("call please");
    }
    Instr::BuiltIn_print_int(arg) => {
        let arg: i128 = read_stack(stack, *arg);
        crate::builtin::print_int(arg);
    }
    Instr::BuiltIn_print_uint(arg) => {
        let arg: u128 = read_stack(stack, *arg);
        crate::builtin::print_uint(arg);
    }
    Instr::BuiltIn_print_float(arg) => {
        let arg: f64 = read_stack(stack, *arg);
        crate::builtin::print_float(arg);
    }
    Instr::BuiltIn_print_bool(arg) => {
        let arg: bool = read_stack(stack, *arg);
        crate::builtin::print_bool(arg);
    }
    Instr::BuiltIn_print_char(arg) => {
        let arg: char = read_stack(stack, *arg);
        crate::builtin::print_char(arg);
    }
    _ => panic!("NYI {:?}",instr)
}"#,
    );

    std::fs::write("src/vm/_exec_match.txt", source).unwrap();
}
