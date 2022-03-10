use iced_x86::{Decoder,DecoderOptions,IntelFormatter,Formatter,Instruction};

pub fn disassemble(code: &[u8]) {
    
    let mut decoder = Decoder::new(64, code, DecoderOptions::NONE);
    decoder.set_ip(0x1000);

    let mut formatter = IntelFormatter::new();

    let mut instruction = Instruction::default();
    let mut output = String::new();
    while decoder.can_decode() {
        output.clear();
        decoder.decode_out(&mut instruction);
        formatter.format(&instruction, &mut output);
        println!("  {:02x}  {}",instruction.ip(),output);
    }
}
