pub extern "C" fn print_i64(x: i64) {
    println!("{}",x);
}

pub extern "C" fn print_f64(x: f64) {
    println!("{}",x);
}

pub extern "C" fn print_header(x: i64) {
    println!("> {}",x);
}

pub extern "C" fn assert(index: i32, val: bool) {
    if !val {
        panic!("assert #{} failed",index);
    }   
}
