mod _skitter_builtin;

fn main() {
    let x = 90u8;

    _skitter_builtin::print_i64(x as _);
    _skitter_builtin::print_f64(x as _);
    _skitter_builtin::print_char(x as _);
}
