mod _skitter_builtin;

fn f(x: (), y: (), z:(), n: f64, m: i64) -> () {
    _skitter_builtin::print_f64(n);
    _skitter_builtin::print_i64(m);
}

fn main() {
    let res: () = if true {
        f((),(),(),100.0,200);
    };
}
