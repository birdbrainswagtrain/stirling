mod _builtin;

fn print_float_ref(x: &f64) {
    _builtin::print_float(*x);
}

fn edit_float_ref(x: &mut f64) {
    *x = *x + 10.0;
}

// todo += edit
// todo ref in assign
// todo assign to mut ref

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
