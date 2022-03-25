mod _builtin;

pub fn main() {
    let x: i32 = 1_528_267_100;
    let y: i32 = -66;
    
    _builtin::print_i32(-x);
    _builtin::print_i32(x + y);
    _builtin::print_i32(x - y);
    _builtin::print_i32(x * y);
    _builtin::print_i32(x / y);
    _builtin::print_i32(x % y);

    _builtin::print_i32(!x);
    _builtin::print_i32(x & y);
    _builtin::print_i32(x | y);
    _builtin::print_i32(x ^ y);

    _builtin::print_bool(x == y);
    _builtin::print_bool(x != y);
    _builtin::print_bool(x < y);
    _builtin::print_bool(x > y);
    _builtin::print_bool(x <= y);
    _builtin::print_bool(x >= y);
}
