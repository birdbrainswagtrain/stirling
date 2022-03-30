mod _builtin;

pub fn main() {
    let mut i = 0;
    while i < 100 {
        i += 3;
    }
    _builtin::print_int(i);
}
