pub fn add(x: i32, y: i32) -> i32 {
    let z = if x > y { 100 } else { -100 };
    z = z + 10000;
    x + y + z
}
