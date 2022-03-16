mod _skitter_builtin;

fn fail() -> i32 {
    panic!("fail");
}

fn get_n(n: i32) -> i32 {
    if n == 0 {
        0
    } else if n == 1 {
        1
    } else if n == 2 {
        -1
    } else if n == 3 {
        2147483647
    } else if n == 4 {
        -2147483648
    } else if n == 5 {
        523182
    } else if n == 6 {
        -882781
    } else {
        fail()
    }
}

fn main() {
    
    // Addition
    {
        let op = 1;
        let mut i = 0;
        while i < 7 {
            let mut j = 0;
            while j < 7 {
                let a = get_n(i);
                let b = get_n(j);
                _skitter_builtin::print_header(a as i64);
                _skitter_builtin::print_header(op as i64);
                _skitter_builtin::print_header(b as i64);
                _skitter_builtin::print_i64((a+b) as i64);
                j += 1;
            }
            i += 1;
        }
    }

    // Subtraction
    {
        let op = 2;
        let mut i = 0;
        while i < 7 {
            let mut j = 0;
            while j < 7 {
                let a = get_n(i);
                let b = get_n(j);
                _skitter_builtin::print_header(a as i64);
                _skitter_builtin::print_header(op as i64);
                _skitter_builtin::print_header(b as i64);
                _skitter_builtin::print_i64((a-b) as i64);
                j += 1;
            }
            i += 1;
        }
    }

    // Multiply
    {
        let op = 3;
        let mut i = 0;
        while i < 7 {
            let mut j = 0;
            while j < 7 {
                let a = get_n(i);
                let b = get_n(j);
                _skitter_builtin::print_header(a as i64);
                _skitter_builtin::print_header(op as i64);
                _skitter_builtin::print_header(b as i64);
                _skitter_builtin::print_i64((a*b) as i64);
                j += 1;
            }
            i += 1;
        }
    }

    // Divide
    {
        let op = 4;
        let mut i = 0;
        while i < 7 {
            let mut j = 1; // SKIP ZERO
            while j < 7 {
                let a = get_n(i);
                let b = get_n(j);
                if (i == 4) & (j == 2) { // SKIP MIN / -1
                    j += 1;
                    continue;
                }
                _skitter_builtin::print_header(a as i64);
                _skitter_builtin::print_header(op as i64);
                _skitter_builtin::print_header(b as i64);
                _skitter_builtin::print_i64((a/b) as i64);
                j += 1;
            }
            i += 1;
        }
    }

    // Modulo
    {
        let op = 5;
        let mut i = 0;
        while i < 7 {
            let mut j = 1; // SKIP ZERO
            while j < 7 {
                let a = get_n(i);
                let b = get_n(j);
                if (i == 4) & (j == 2) { // SKIP MIN / -1
                    j += 1;
                    continue;
                }
                _skitter_builtin::print_header(a as i64);
                _skitter_builtin::print_header(op as i64);
                _skitter_builtin::print_header(b as i64);
                _skitter_builtin::print_i64((a/b) as i64);
                j += 1;
            }
            i += 1;
        }
    }

    // Or
    {
        let op = 11;
        let mut i = 0;
        while i < 7 {
            let mut j = 0;
            while j < 7 {
                let a = get_n(i);
                let b = get_n(j);
                _skitter_builtin::print_header(a as i64);
                _skitter_builtin::print_header(op as i64);
                _skitter_builtin::print_header(b as i64);
                _skitter_builtin::print_i64((a|b) as i64);
                j += 1;
            }
            i += 1;
        }
    }

    // And
    {
        let op = 12;
        let mut i = 0;
        while i < 7 {
            let mut j = 0;
            while j < 7 {
                let a = get_n(i);
                let b = get_n(j);
                _skitter_builtin::print_header(a as i64);
                _skitter_builtin::print_header(op as i64);
                _skitter_builtin::print_header(b as i64);
                _skitter_builtin::print_i64((a&b) as i64);
                j += 1;
            }
            i += 1;
        }
    }

    // Xor
    {
        let op = 13;
        let mut i = 0;
        while i < 7 {
            let mut j = 0;
            while j < 7 {
                let a = get_n(i);
                let b = get_n(j);
                _skitter_builtin::print_header(a as i64);
                _skitter_builtin::print_header(op as i64);
                _skitter_builtin::print_header(b as i64);
                _skitter_builtin::print_i64((a^b) as i64);
                j += 1;
            }
            i += 1;
        }
    }
}
