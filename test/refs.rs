mod _skitter_builtin;

fn print_int_ref(x: &i32) {
    _skitter_builtin::print_i64(*x as i64);
}

fn edit_int_ref(x: &mut i32) {
    *x = 100;
}

fn main() {
    let x = 10;
    let y = &x;
    print_int_ref(&x);
    print_int_ref(y);
    print_int_ref(&20);

    {
        let mut m = 0;
        edit_int_ref(&mut m);
        print_int_ref(&m);
    }

    {
        let mut m = 0;
        edit_int_ref(&mut * &mut m);
        print_int_ref(& * & m);
    }
}
