use crate::pim::{operation::PimOperand, vector::F16x1};
use nalgebra::SVector;
use pim_isa::{File, Instruction, Kernel};

pub const KERNEL: Kernel = Kernel([
    // 5 instructions actives
    Instruction::MOV { src: File::Bank, dst: File::GrfA { index: 0 } },
    Instruction::MOV { src: File::Bank, dst: File::GrfA { index: 1 } },
    Instruction::MAC { 
        src0: File::Bank,
        src1: File::GrfA { index: 0 },
        src2: File::GrfA { index: 1 },
        dst: File::GrfB { index: 0 },
        aam: false 
    },
    Instruction::FILL { src: File::GrfB { index: 0 }, dst: File::Bank },
    Instruction::EXIT,
    
    // 27 instructions NOP (pour remplir le cache de 32)
    Instruction::NOP, Instruction::NOP, Instruction::NOP,
    Instruction::NOP, Instruction::NOP, Instruction::NOP, Instruction::NOP,
    Instruction::NOP, Instruction::NOP, Instruction::NOP, Instruction::NOP,
    Instruction::NOP, Instruction::NOP, Instruction::NOP, Instruction::NOP,
    Instruction::NOP, Instruction::NOP, Instruction::NOP, Instruction::NOP,
    Instruction::NOP, Instruction::NOP, Instruction::NOP, Instruction::NOP,
    Instruction::NOP, Instruction::NOP, Instruction::NOP, Instruction::NOP,
]);

pub fn execute<
    const NNZ: usize,
    const ROWS: usize,
    const COLS: usize,
    const ROW_PTR_SIZE: usize,
    const BLOCKS: usize
>(
    values: &SVector<F16x1, NNZ>,
    col_indices: &SVector<F16x1, NNZ>, 
    row_ptrs: &SVector<F16x1, ROW_PTR_SIZE>,
    x: &SVector<F16x1, COLS>,
    y: &mut SVector<F16x1, ROWS>,
    dummy: &impl PimOperand,
) {
    values.fixed_rows_with_step::<BLOCKS>(0, 256).iter().for_each(|entry| entry.execute_read());
    col_indices.fixed_rows_with_step::<BLOCKS>(0, 256).iter().for_each(|entry| entry.execute_read());
    row_ptrs.fixed_rows_with_step::<BLOCKS>(0, 256).iter().for_each(|entry| entry.execute_read());
    x.fixed_rows_with_step::<BLOCKS>(0, 256).iter().for_each(|entry| entry.execute_read());
    y.fixed_rows_with_step_mut::<BLOCKS>(0, 256).iter_mut().for_each(|entry| entry.execute_write());

    dummy.execute_read();
}
