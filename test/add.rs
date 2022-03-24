pub fn main(a: i32, b: i32) -> i32 {
    let mut i = 0;
    let mut sum = 0;
    while i < 1_000_000_000 {
        i = i + 1;
        sum = sum + 1 + 1;
    }
    sum
}
