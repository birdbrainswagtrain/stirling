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

fn bad() -> bool {
    "it doesn't even support strings" >> lmao
}

fn main() {

    let z = 1.0;

    while z <= 100.0 {
        _skitter_builtin::print_f64(z);
        z += 1.0;
    }
    
    /*
    print_bool('a' == 'a');
    print_bool('a' == 'b');
    _skitter_builtin::print_i32(11111);
    print_bool('A' > 'a');
    print_bool('A' == 'a');
    print_bool('A' < 'a');*/
}
