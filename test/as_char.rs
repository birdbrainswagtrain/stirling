mod _skitter_builtin;

fn main() {
    {
        let x = '😁';
        let a = x as u64;
        let b = x as i8;
        _skitter_builtin::print_i64(a as i64);
        _skitter_builtin::print_i64(b as i64);
    }

    {
        let x = 0xD7u8;
        _skitter_builtin::print_char(x as char);
    }
}
