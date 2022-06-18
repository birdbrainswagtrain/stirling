mod _builtin;

fn print_float_ref(x: &f64) {
    _builtin::print_float(*x);
}

fn edit_float_ref(x: &mut f64) {
    *x = *x + 10.0;
}

fn edit_float_ref_2(x: &mut f64) {
    *x += 1.0;
}

// todo ref in assign

fn main() {
    let x = 5.0;
    let y = &x;
    print_float_ref(&x);
    print_float_ref(y);
    print_float_ref(&25.0);

    (*&mut 10) = 20;

    {
        let mut m = 0.0;
        edit_float_ref(&mut m);
        print_float_ref(&m);
        
        edit_float_ref(&mut * &mut m);
        print_float_ref(& * & m);

        edit_float_ref_2(&mut m);
        print_float_ref(&m);
    }
}
