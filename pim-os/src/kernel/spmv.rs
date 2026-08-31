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

// new update spmv du 29/8/26

pub fn execute_bank<
    const NNZ: usize,
    const ROWS: usize,
    const COLS: usize,
    const ROW_PTR_SIZE: usize,
>(
    values: &SVector<F16x1, NNZ>,
    col_indices: &SVector<F16x1, NNZ>, 
    row_ptrs: &SVector<F16x1, ROW_PTR_SIZE>,
    x: &SVector<F16x1, COLS>,
    y: &mut SVector<F16x1, ROWS>,
    start_row: usize,
    end_row: usize,
    start_nnz: usize,
    end_nnz: usize,
    dummy: &impl PimOperand,
) {
    // Le CPU déclenche les transactions TLM (read/write) uniquement 
    // pour la sous-matrice assignée à cette banque spécifique.
    for i in start_nnz..end_nnz {
        values[i].execute_read();
        col_indices[i].execute_read();
        // Dans une vraie architecture multi-banque, X est diffusé, 
        // nous simulons ici son accès par la banque.
        let col = col_indices[i].0.to_f32() as usize;
        if col < COLS {
            x[col].execute_read();
        }
    }

    for i in start_row..end_row {
        row_ptrs[i].execute_read();
        y[i].execute_write();
    }

    dummy.execute_read();
}
