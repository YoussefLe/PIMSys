#![no_std]
#![no_main]

extern crate alloc;

use aarch64_cpu::asm::barrier;
use alloc::boxed::Box;
use half::f16;
use nalgebra::SVector;
use pim_os::pim::vector::F16x1;

const ROWS: usize = 2048;
const COLS: usize = 2048;
const NNZ: usize = 8192;
const ROW_PTR_SIZE: usize = ROWS + 1;

#[no_mangle]
pub extern "C" fn main() {
    // 1. Allocation identique
    let mut values = Box::new(SVector::<F16x1, NNZ>::zeros());
    let mut col_indices = Box::new(SVector::<F16x1, NNZ>::zeros());
    let mut row_ptrs = Box::new(SVector::<F16x1, ROW_PTR_SIZE>::zeros());
    let mut x = Box::new(SVector::<F16x1, COLS>::zeros());
    let mut y = Box::new(SVector::<F16x1, ROWS>::zeros());

    // 2. Génération de la matrice identique
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

    barrier::dsb(barrier::SY);

    // 3. --- DÉBUT DU CALCUL CPU (SANS PIM) ---
    for i in 0..ROWS {
        let start = row_ptrs[i].0.to_f32() as usize;
        let end = row_ptrs[i + 1].0.to_f32() as usize;
        
        let mut sum = 0.0;
        for j in start..end {
            let val = values[j].0.to_f32();
            let col = col_indices[j].0.to_f32() as usize;
            sum += val * x[col].0.to_f32();
        }
        y[i] = F16x1(f16::from_f32(sum));
    }
    // --- FIN DU CALCUL CPU ---

    barrier::dsb(barrier::SY);

    loop {
        aarch64_cpu::asm::wfi();
    }
}
