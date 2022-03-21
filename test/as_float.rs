mod _skitter_builtin;

fn main() {

    // f32 to f64
    {
        let a: f32 = 1042.5213;
        let b: f64 = 1042.5213;
        let inf: f32 = 1.0/0.0;
        let nan: f32 = 0.0/0.0;

        _skitter_builtin::print_f64(a as f64);
        _skitter_builtin::print_f64(b);
        _skitter_builtin::print_f64(inf as f64);
        _skitter_builtin::print_f64(nan as f64);
    }

    // f64 to f32
    {
        let a: f64 = 16777216.0;
        let b: f64 = 16777217.0;
        let c: f64 = 16777218.0;

        _skitter_builtin::print_f64(a as f32 as f64);
        _skitter_builtin::print_f64(b as f32 as f64);
        _skitter_builtin::print_f64(c as f32 as f64);

        _skitter_builtin::print_f64(-a as f32 as f64);
        _skitter_builtin::print_f64(-b as f32 as f64);
        _skitter_builtin::print_f64(-c as f32 as f64);

        let inf: f64 = 1.0/0.0;
        let nan: f64 = 0.0/0.0;
        _skitter_builtin::print_f64(inf as f32 as f64);
        _skitter_builtin::print_f64(nan as f32 as f64);

        let overflows: f64 = 1e100;
        _skitter_builtin::print_f64(overflows);
        _skitter_builtin::print_f64(overflows as f32 as f64);
    }

    // int -> float
    {
        let x = 100;
        _skitter_builtin::print_f64(x as f32 as f64);
        _skitter_builtin::print_f64(x as f64);
        _skitter_builtin::print_f64(-x as f32 as f64);
        _skitter_builtin::print_f64(-x as f64);
        _skitter_builtin::print_f64(-x as u8 as f32 as f64);
        _skitter_builtin::print_f64(-x as u64 as f64);

        let a = 16777216;
        let b = 16777217;
        let c = 16777218;
    
        _skitter_builtin::print_f64(a as f32 as f64);
        _skitter_builtin::print_f64(b as f32 as f64);
        _skitter_builtin::print_f64(c as f32 as f64);

        _skitter_builtin::print_f64(-a as f32 as f64);
        _skitter_builtin::print_f64(-b as f32 as f64);
        _skitter_builtin::print_f64(-c as f32 as f64);
    }

    // float -> int
    {
        let z = 0.0;
        let a = 5.5;
        let b = -5.5;
        let c = 5.7;
        let d = -5.7;
        let e = 5_000_000_000.0;
        let f = -5_000_000_000.0;
        let inf = 1.0/0.0;
        let nan = 0.0/0.0;

        _skitter_builtin::print_i64(z as i32 as i64);
        _skitter_builtin::print_i64(a as i32 as i64);
        _skitter_builtin::print_i64(b as i32 as i64);
        _skitter_builtin::print_i64(c as i32 as i64);
        _skitter_builtin::print_i64(d as i32 as i64);
        _skitter_builtin::print_i64(e as i32 as i64);
        _skitter_builtin::print_i64(f as i32 as i64);
        _skitter_builtin::print_i64(inf as i32 as i64);
        _skitter_builtin::print_i64(-inf as i32 as i64);
        _skitter_builtin::print_i64(nan as i32 as i64);

        _skitter_builtin::print_i64(z as u32 as i64);
        _skitter_builtin::print_i64(a as u32 as i64);
        _skitter_builtin::print_i64(b as u32 as i64);
        _skitter_builtin::print_i64(c as u32 as i64);
        _skitter_builtin::print_i64(d as u32 as i64);
        _skitter_builtin::print_i64(e as u32 as i64);
        _skitter_builtin::print_i64(f as u32 as i64);
        _skitter_builtin::print_i64(inf as u32 as i64);
        _skitter_builtin::print_i64(-inf as u32 as i64);
        _skitter_builtin::print_i64(nan as u32 as i64);

        _skitter_builtin::print_i64(z as i8 as i64);
        _skitter_builtin::print_i64(a as i8 as i64);
        _skitter_builtin::print_i64(b as i8 as i64);
        _skitter_builtin::print_i64(c as i8 as i64);
        _skitter_builtin::print_i64(d as i8 as i64);
        _skitter_builtin::print_i64(e as i8 as i64);
        _skitter_builtin::print_i64(f as i8 as i64);
        _skitter_builtin::print_i64(inf as i8 as i64);
        _skitter_builtin::print_i64(-inf as i8 as i64);
        _skitter_builtin::print_i64(nan as i8 as i64);

        _skitter_builtin::print_i64(z as u8 as i64);
        _skitter_builtin::print_i64(a as u8 as i64);
        _skitter_builtin::print_i64(b as u8 as i64);
        _skitter_builtin::print_i64(c as u8 as i64);
        _skitter_builtin::print_i64(d as u8 as i64);
        _skitter_builtin::print_i64(e as u8 as i64);
        _skitter_builtin::print_i64(f as u8 as i64);
        _skitter_builtin::print_i64(inf as u8 as i64);
        _skitter_builtin::print_i64(-inf as u8 as i64);
        _skitter_builtin::print_i64(nan as u8 as i64);
    }
}
