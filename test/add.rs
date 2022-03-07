pub fn add(x: i32, y: i32) -> i32 {
    let m = true;
    let n = false;
    let mut z = 500 + if x > y { 100 } else { -100 } + 500;
    let o = (x * y) / 2;
    /*let mut z = {
        let m = 10;
        m + 90
    };*/
    //z += 10000;
    z
}
