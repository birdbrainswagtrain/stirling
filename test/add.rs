mod _builtin;

pub fn main() {
    let mut i = 0;
    let mut sum = 0;
    while i < 10_000 {
        i = i + 1;
        sum = (sum + 25/5 - 2*2) % 123;
    }
    _builtin::print_i32(sum);
}
