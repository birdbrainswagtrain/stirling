mod _skitter_builtin;

fn main() {
    let x = true;
    let y = false;

    _skitter_builtin::print_i64(x as i64);
    _skitter_builtin::print_i64(y as i64);
}
