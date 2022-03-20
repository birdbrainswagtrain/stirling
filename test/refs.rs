mod _skitter_builtin;

fn print_float_ref(x: &f64) {
    _skitter_builtin::print_f64(*x);
}

fn edit_float_ref(x: &mut f64) {
    *x += 10.0;
}

fn main() {
    let x = 5.0;
    let y = &x;
    print_float_ref(&x);
    print_float_ref(y);
    print_float_ref(&25.0);

    {
        let mut m = 0.0;
        edit_float_ref(&mut m);
        print_float_ref(&m);
        
        edit_float_ref(&mut * &mut m);
        print_float_ref(& * & m);
    }
}
