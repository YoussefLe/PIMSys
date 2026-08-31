#![no_std]
#![no_main]

extern crate alloc;

use aarch64_cpu::asm::barrier;
use alloc::boxed::Box;
use half::f16;
use nalgebra::SVector;
use pim_os::{kernel::spmv, pim::vector::F16x1};
use pim_isa::BankMode;

const ROWS: usize = 2048;
const COLS: usize = 2048;
const NNZ: usize = 8192;
const ROW_PTR_SIZE: usize = ROWS + 1;
const BLOCKS: usize = 8;
const NUM_BANKS: usize = 32;

// Fonction de Load Balancing exécutée par le CPU ARM
fn partition_csr(row_ptrs: &SVector<F16x1, ROW_PTR_SIZE>) -> [usize; NUM_BANKS + 1] {
    let mut bounds = [0; NUM_BANKS + 1];
    let target_nnz_per_bank = NNZ / NUM_BANKS;
    let mut current_bank = 1;
    
    for i in 0..ROWS {
        let current_nnz = row_ptrs[i].0.to_f32() as usize;
        if current_nnz >= target_nnz_per_bank * current_bank && current_bank < NUM_BANKS {
            bounds[current_bank] = i;
            current_bank += 1;
        }
    }
    bounds[NUM_BANKS] = ROWS;
    bounds
}

#[no_mangle]
pub extern "C" fn main() {
    pim_os::pim::state::set_kernel(&spmv::KERNEL);

    // 1. Allocation (inchangée)
    let mut values = Box::new(SVector::<F16x1, NNZ>::zeros());
    let mut col_indices = Box::new(SVector::<F16x1, NNZ>::zeros());
    let mut row_ptrs = Box::new(SVector::<F16x1, ROW_PTR_SIZE>::zeros());
    let mut x = Box::new(SVector::<F16x1, COLS>::zeros());
    let mut y = Box::new(SVector::<F16x1, ROWS>::zeros());

    // 2. Génération de la matrice (inchangée)
    for i in 0..ROWS {
        row_ptrs[i] = F16x1(f16::from_f32((i * 4) as f32));
        for k in 0..4 {
            let idx = i * 4 + k;
            values[idx] = F16x1(f16::from_f32(1.5));
            col_indices[idx] = F16x1(f16::from_f32(((i + k) % COLS) as f32));
        }
        x[i] = F16x1(f16::from_f32(i as f32));
    }
    row_ptrs[ROWS] = F16x1(f16::from_f32(NNZ as f32));

    // 3. Phase HPC : Équilibrage de charge
    let bank_row_bounds = partition_csr(&row_ptrs);
    barrier::dsb(barrier::SY);

    let dummy = F16x1(f16::from_f32(0.0));

    // 4. Dispatch aux 32 banques
    for bank_id in 0..NUM_BANKS {
        let start_row = bank_row_bounds[bank_id];
        let end_row = bank_row_bounds[bank_id + 1];
        
        let start_nnz = row_ptrs[start_row].0.to_f32() as usize;
        let end_nnz = row_ptrs[end_row].0.to_f32() as usize;

        // Ici, en matériel réel, on configurerait les registres MMIO de chaque banque.
        // Dans notre simulateur, on force les lectures TLM sur les plages spécifiques.
        spmv::execute_bank(
            &values, &col_indices, &row_ptrs, &x, &mut y, 
            start_row, end_row, start_nnz, end_nnz, &dummy
        );
    }

    barrier::dsb(barrier::SY);
    
    // Le CPU s'endort pendant que la HBM2 calcule en parallèle
    loop {
        aarch64_cpu::asm::wfi();
    }
}
