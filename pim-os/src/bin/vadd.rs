#![no_std]
#![no_main]

extern crate alloc;

use aarch64_cpu::asm::barrier;
use alloc::boxed::Box;
use half::f16;
use nalgebra::SVector;
use pim_os::{kernel::spmv, pim::vector::F16x1};
use pim_isa::BankMode; // <-- La correction est ici !

const ROWS: usize = 2048;
const COLS: usize = 2048;
const NNZ: usize = 8192;
const ROW_PTR_SIZE: usize = ROWS + 1;
const BLOCKS: usize = 8;

#[no_mangle]
pub extern "C" fn main() {
    pim_os::pim::state::set_kernel(&spmv::KERNEL);

    let mut values = Box::new(SVector::<F16x1, NNZ>::zeros());
    let mut col_indices = Box::new(SVector::<F16x1, NNZ>::zeros());
    let mut row_ptrs = Box::new(SVector::<F16x1, ROW_PTR_SIZE>::zeros());
    let mut x = Box::new(SVector::<F16x1, COLS>::zeros());
    let mut y = Box::new(SVector::<F16x1, ROWS>::zeros());

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

    pim_os::pim::state::set_bank_mode(BankMode::PimAllBank);

    let dummy = F16x1(f16::from_f32(0.0));
    
    spmv::execute::<NNZ, ROWS, COLS, ROW_PTR_SIZE, BLOCKS>(
        &values,
        &col_indices,
        &row_ptrs,
        &x,
        &mut y,
        &dummy
    );

    barrier::dsb(barrier::SY);
    pim_os::pim::state::set_bank_mode(BankMode::SingleBank);

    loop {
        aarch64_cpu::asm::wfi();
    }
}
