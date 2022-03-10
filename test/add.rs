mod _skitter_builtin;

fn add(x: i32, y: i32) -> i32 {
    let mut sum = 0;
    let mut i = 0;
    while i < y {
        i += 1;
        sum += x;
        sum /= 2;
    };
    sum
}

fn bad() {
    "it doesn't even support strings" >> lmao
}

fn main() {
    let res1 = 100 + 50 + 1;
    _skitter_builtin::print_i32(res1);
    let res2 = add(1000,100_000_000);
    _skitter_builtin::print_i32(res2);
    bad();
}
