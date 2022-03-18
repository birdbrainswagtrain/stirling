mod _skitter_builtin;

fn print_int_ref(x: &i32) {
    _skitter_builtin::print_i64(*x as i64);
}

fn main() {
    let x = 10;
    let y = &x;
    print_int_ref(&x);
    print_int_ref(y);
    print_int_ref(&20);
}
