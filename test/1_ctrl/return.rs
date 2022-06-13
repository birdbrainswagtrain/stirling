mod _builtin;

fn e() -> i32 {
    let x: f32 = {
        return 10;
    };
}

fn f() -> i32 {
    10
}

fn g() -> i32 {
    return 20;
}

fn h() -> i32 {
    {
        {
            return 30;
        };
    };
}

fn i() -> i32 {
    if true {
        return 40;
    } else {
        return -1;
    }
}

fn j() -> i32 {
    if false {
        return -1;
    } else {
        50
    }
}

pub fn main() {
    _builtin::print_int(e() as _);
    //_builtin::print_int(f() as _);
    //_builtin::print_int(g() as _);
    //_builtin::print_int(h() as _);
    //_builtin::print_int(i() as _);
    //_builtin::print_int(j() as _);
}
