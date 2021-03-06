pub extern "C" fn print_int(x: i128) {
    println!("{}", x);
}

pub extern "C" fn print_uint(x: u128) {
    println!("{}", x);
}

pub extern "C" fn print_float(x: f64) {
    println!("{}", x);
}

pub extern "C" fn print_char(x: char) {
    println!("{}", x);
}

pub extern "C" fn print_bool(x: bool) {
    println!("{}", x);
}
