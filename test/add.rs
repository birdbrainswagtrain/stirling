pub fn add(x: i32, y: i32) -> i32 {
    //(x + y) * 2
    let z = 0x10 + 2;
    x + (x + y) * z
    
    // 0 Add(Local(0),Local(1))     i32
    // 1 ImmInt(2,?)                i32
    // 2 Mul(Res(0),Res(1))         i32
    // 3 Block([],Res(2))           i32
}
