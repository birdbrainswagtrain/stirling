mod _skitter_builtin;

pub fn main() {
    let x = (100,10.0);
    _skitter_builtin::print_i64(x.0);
    _skitter_builtin::print_f64(x.1);
}
