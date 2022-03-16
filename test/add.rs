mod _skitter_builtin;

fn print_bool(a: bool) {
    if a {
        _skitter_builtin::print_i32(1);
    } else {
        _skitter_builtin::print_i32(0);
    }
}

fn bit_test(a: bool, b: bool) {
    let res = a ^ b;
    if res {
        _skitter_builtin::print_i32(1);
    } else {
        _skitter_builtin::print_i32(0);
    }
}

fn int_test(a: u32, b: u32) {
    _skitter_builtin::print_i32((a & b) as i32);
    _skitter_builtin::print_i32((a | b) as i32);
    _skitter_builtin::print_i32((a ^ b) as i32);
}

fn tiny(x: i32, y: i32) -> i32 {
    (x + y) / 2
}

fn call_tiny() {
    _skitter_builtin::print_i32(456);
    let mut i = 0;
    let mut sum = 0;
    while i < 1_000_000_000 {
        if sum > 1_000_000_000 {
            sum /= 2;
        } else {
            sum += 100;
        }
        sum += tiny(100,i);
        i += 1;
    }
    _skitter_builtin::print_i32(i);
    _skitter_builtin::print_i32(sum)
}

fn never() -> ! {
    panic!("butt")
}

fn main() {

    /*let mut z = 1.0;

    let res = 'a: loop {
        if z > 3.0 {
            break 'a 6023;
        }
        _skitter_builtin::print_f64(z);
        z += 0.1;
    };
    _skitter_builtin::print_i32(res);*/

    /*let mut i = 0;
    let res = loop {
        i += 1;
        if i > 100 {
            break 1234;
        }
        //_skitter_builtin::print_i32(5);
    };

    _skitter_builtin::print_i32(10);
    _skitter_builtin::print_i32(res);*/

    let mut i = 1;
    let x: i32 = loop {
        i += 1;
    };

    _skitter_builtin::print_i32(x);
}
