mod _skitter_builtin;

fn add(x: i32, y: i32) -> i32 {
    let mut i = 0;
    let mut sum = 0;
    while i < y {
        i += 1;
        sum += x;
        sum /= 2;
    };
    sum
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

fn bad() {
    "it doesn't even support strings" >> lmao
}

fn main() {
    _skitter_builtin::print_i32(123);
    call_tiny();
    _skitter_builtin::print_i32(789);
    bad();
}
