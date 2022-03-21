fn main() {
    {
        let a = 100;
        let b = &a;
        let c = { &*b };
        let d = &{ *b };
    
        let pb = b as *const i32 as usize;
        let pc = c as *const i32 as usize;
        let pd = d as *const i32 as usize;

        println!("2 {}",pb == pc);
        println!("2 {}",pc != pd);
    }
}
