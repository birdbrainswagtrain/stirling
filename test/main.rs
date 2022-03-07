use std::ops::Add;

impl Add for u8 {
    //type Rhs = u32;
    type Output = u64;

    fn add(self, rhs: u32) -> Self::Output {
        panic!("stop");
    }
}

fn main() {
    
    let a = 100u8;
    let b = 50u32;

    let x = a + b;

    print_type_of(a);
    print_type_of(b);
    print_type_of(x);
}

fn print_type_of<T>(_: T) {
    println!("{}", std::any::type_name::<T>())
}
