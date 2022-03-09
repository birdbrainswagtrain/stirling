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

fn main() {
    let res = add(10000,100_000_000);
    println!("{}",res);
}
