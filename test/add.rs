mod _skitter_builtin;

fn add(x: i32, y: i32) -> i32 {
    let mut sum = 0;
    let mut i = 0;
    while i < y {
        i += 1;
        sum += x;
        sum /= 2;
    }
    sum
}

fn main() {
    //let res = add(10000,100_000_000);
    let res = 100 + 50 + 1;
    _skitter_builtin::print_i32(res);
    _skitter_builtin::print_i32(521);
    _skitter_builtin::print_i32(222);
    _skitter_builtin::print_i32(1001);
}
