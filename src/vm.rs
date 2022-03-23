use crate::{hir::{
    func::{Block, Expr, ExprInfo, FuncHIR},
    item::Function,
    types::{Type, IntType},
}, profiler::profile};

#[derive(Clone, Copy, Default)]
struct FrameAllocator(u32);

impl FrameAllocator {
    fn alloc(&mut self, ty: Type) -> Option<u32> {
        let align = ty.byte_align() as u32;
        let size = ty.byte_size() as u32;
        while self.0 % align != 0 {
            self.0 += 1;
        }
        let offset = self.0;
        self.0 += size;
        Some(offset)
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug,Copy,Clone)]
enum Instr {
    // out = n
    I32_Const(u32, i32),
    // out = lhs % rhs
    I32_Add(u32, u32, u32),
    Return
}

unsafe fn set_i32(base: *mut u8, offset: u32, n: i32) {
    *(base.add(offset as usize) as *mut i32) = n;
}

unsafe fn get_i32(base: *mut u8, offset: u32) -> i32 {
    *(base.add(offset as usize) as *mut i32)
}

pub fn exec(func: &Function, args: &[i32]) -> i32 {
    let code = profile("lower HIR -> BC", || {
        compile(func)
    });

    let mut stack: Vec<u128> = vec![0, 1024];
    profile("exec",|| unsafe {
        let stack = stack.as_mut_ptr() as *mut u8;

        let mut pc = 0;
    
        // jank arg setup
        for (i,arg) in args.iter().enumerate() {
            set_i32(stack, (4 + i*4) as u32, *arg);
        }
        
        loop {
            let instr = code[pc];
            match instr {
                Instr::I32_Add(out, lhs, rhs) => {
                    let n = get_i32(stack, lhs) + get_i32(stack, rhs);
                    set_i32(stack, out, n);
                },
                Instr::I32_Const(out, n) => {
                    set_i32(stack, out, n);
                }
                Instr::Return => break,
                _ => panic!("todo execute {:?}",instr)
            }
            pc += 1;
        }

        // jank returns
        get_i32(stack,0)
    })
}

fn compile(func: &Function) -> Vec<Instr> {
    let sig = func.sig();
    let input_fn = func.hir();

    let mut frame = FrameAllocator::default();
    let return_slot = frame.alloc(sig.output);
    
    match return_slot {
        Some(offset) => {
            assert_eq!(offset,0);
        }
        None => (),
    }

    let var_map: Vec<_> = input_fn
        .vars
        .iter()
        .map(|var| {
            let ty = input_fn.exprs[*var as usize].ty;
            frame.alloc(ty)
        })
        .collect();

    let mut compiler = BCompiler {
        var_map,
        code: Vec::new(),
        input_fn,
        frame,
    };

    compiler.lower_expr(input_fn.root_expr as u32, return_slot);
    compiler.code.push(Instr::Return);
    compiler.code
}

struct BCompiler<'a> {
    var_map: Vec<Option<u32>>,
    code: Vec<Instr>,
    input_fn: &'a FuncHIR,
    frame: FrameAllocator,
}

fn instr_for_bin_op(op: syn::BinOp, ty: Type) -> fn(u32, u32, u32) -> Instr {
    Instr::I32_Add
}

impl<'a> BCompiler<'a> {
    fn lower_expr(&mut self, expr_id: u32, result_slot: Option<u32>) -> Option<u32> {
        let mut frame = self.frame;

        let ExprInfo { expr, ty, .. } = &self.input_fn.exprs[expr_id as usize];
        match expr {
            Expr::Var(id) => self.var_map[*id as usize],
            Expr::LitInt(n) => {
                if let Some(result_slot) = result_slot {
                    match ty {
                        Type::Int(IntType::I32) | Type::Int(IntType::U32) => {
                            self.code.push(Instr::I32_Const(result_slot, *n as i32));
                        }
                        _ => panic!("todo more literal ints")
                    }
                }
                result_slot
            }
            Expr::BinOpPrimitive(lhs, op, rhs) => {
                let l_slot = frame.alloc(*ty);
                let lhs = self.lower_expr(*lhs, l_slot).unwrap();
                let r_slot = frame.alloc(*ty);
                let rhs = self.lower_expr(*rhs, r_slot).unwrap();
                let ins_ctor = instr_for_bin_op(*op,*ty);
                let ins = ins_ctor(result_slot.unwrap(),lhs,rhs);
                self.code.push(ins);
                result_slot
            }
            Expr::Block(block) => self.lower_block(block, result_slot),
            _ => panic!("vm compile {:?}", expr),
        }
    }

    fn lower_block(&mut self, block: &Block, result_slot: Option<u32>) -> Option<u32> {
        for expr_id in &block.stmts {
            self.lower_expr(*expr_id, None);
        }

        if let Some(expr_id) = &block.result {
            self.lower_expr(*expr_id, result_slot)
        } else {
            None
        }
    }
}
