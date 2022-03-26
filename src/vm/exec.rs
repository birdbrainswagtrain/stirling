use super::Instr;

unsafe fn write_stack<T>(base: *mut u8, offset: u32, x: T) {
    *(base.add(offset as usize) as *mut _) = x;
}

unsafe fn read_stack<T: Copy>(base: *mut u8, offset: u32) -> T {
    *(base.add(offset as usize) as *mut _)
}

pub unsafe fn exec_rust(code: Vec<Instr>, stack: *mut u8) {
    let mut pc = 0;

    loop {
        let instr = code[pc]; //*code.get_unchecked(pc);// [pc];
        include!("_exec_match.txt");
        pc += 1;
    }
}
