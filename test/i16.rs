mod _builtin;

pub fn main() {
    {
        let x: i16 = 30123;
        let y: i16 = -5201;
        
        _builtin::print_i64((-x) as _);
        _builtin::print_i64((x + y) as _);
        _builtin::print_i64((x - y) as _);
        _builtin::print_i64((x * y) as _);
        _builtin::print_i64((x / y) as _);
        _builtin::print_i64((x % y) as _);
    
        _builtin::print_i64((!x) as _);
        _builtin::print_i64((x & y) as _);
        _builtin::print_i64((x | y) as _);
        _builtin::print_i64((x ^ y) as _);
        _builtin::print_i64((x >> 4) as _);
        _builtin::print_i64((y >> 4) as _);
        _builtin::print_i64((x << 4) as _);
        _builtin::print_i64((y << 4) as _);
    
        _builtin::print_bool(x == y);
        _builtin::print_bool(x != y);
        _builtin::print_bool(x < y);
        _builtin::print_bool(x > y);
        _builtin::print_bool(x <= y);
        _builtin::print_bool(x >= y);

        let mut m = x;
        m += y; _builtin::print_i64(m as _);
        m -= y; _builtin::print_i64(m as _);
        m *= y; _builtin::print_i64(m as _);
        m /= y; _builtin::print_i64(m as _);
        m %= y; _builtin::print_i64(m as _);
        m |= x; _builtin::print_i64(m as _);
        m &= y; _builtin::print_i64(m as _);
        m ^= y; _builtin::print_i64(m as _);
        m >>= 4; _builtin::print_i64(m as _);
        m <<= 4; _builtin::print_i64(m as _);
    }

    {
        let x: u16 = 61021;
        let y: u16 = 321;
        
        _builtin::print_u64((x + y) as _);
        _builtin::print_u64((x - y) as _);
        _builtin::print_u64((x * y) as _);
        _builtin::print_u64((x / y) as _);
        _builtin::print_u64((x % y) as _);
    
        _builtin::print_u64((!x) as _);
        _builtin::print_u64((x & y) as _);
        _builtin::print_u64((x | y) as _);
        _builtin::print_u64((x ^ y) as _);
        _builtin::print_u64((x >> 4) as _);
        _builtin::print_u64((y >> 4) as _);
        _builtin::print_u64((x << 4) as _);
        _builtin::print_u64((y << 4) as _);
    
        _builtin::print_bool(x == y);
        _builtin::print_bool(x != y);
        _builtin::print_bool(x < y);
        _builtin::print_bool(x > y);
        _builtin::print_bool(x <= y);
        _builtin::print_bool(x >= y);

        let mut m = x;
        m += y; _builtin::print_u64(m as _);
        m -= y; _builtin::print_u64(m as _);
        m *= y; _builtin::print_u64(m as _);
        m /= y; _builtin::print_u64(m as _);
        m %= y; _builtin::print_u64(m as _);
        m |= x; _builtin::print_u64(m as _);
        m &= y; _builtin::print_u64(m as _);
        m ^= y; _builtin::print_u64(m as _);
        m >>= 4; _builtin::print_u64(m as _);
        m <<= 4; _builtin::print_u64(m as _);
    }
}
