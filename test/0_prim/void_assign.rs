
mod _builtin;

fn main() {
    let mut x = 10;
    let r1 = if true { x = 90 } else { () };
    let r2 = if true { x += 10 } else { () };
    let r3 = if true { x <<= 1 } else { () };

    _builtin::print_int(x);
}