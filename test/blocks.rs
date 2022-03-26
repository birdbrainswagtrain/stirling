mod _builtin;

pub fn main() {
    let x = {
        let a = 5;
        let b = a;
        let c = a;
        let d = a;
        a + b + c + d
    } + ({
        let a = 10;
        let b = a;
        let c = a;
        let d = a;
        a + b + c + d
    } + {
        let a = 100;
        let b = a;
        let c = a;
        let d = a;
        a + b + c + d
    });
    _builtin::print_i32(x);

    let y = { 11 + 100 };
    _builtin::print_i32(y);

    _builtin::print_i32({1 + 2 + 4 + 8});
}
