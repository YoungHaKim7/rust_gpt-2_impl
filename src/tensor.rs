// the memory machinery that lets many tensors share one big allocation:
// defines: TensorView, split_disjoint

/// a (start, len) view of one tensor inside the single big allocation,
/// the safe equivalent of the C code's pointers into params_memory / acts_memory
#[derive(Clone, Copy, Debug)]
pub struct TensorView {
    pub start: usize,
    pub len: usize,
}

impl TensorView {
    pub const EMPTY: TensorView = TensorView { start: 0, len: 0 };

    /// the (start, len) range of this tensor at `offset` elements in, `len` elements long
    pub fn range(&self, offset: usize, len: usize) -> (usize, usize) {
        (self.start + offset, len)
    }

    /// a shared slice of this tensor at `offset` elements in, `len` elements long
    pub fn slice<'a>(&self, buf: &'a [f32], offset: usize, len: usize) -> &'a [f32] {
        &buf[self.start + offset..self.start + offset + len]
    }
}

/// carve N disjoint (start, len) ranges out of one buffer as mutable slices;
/// this is what lets us keep the C code's "point many tensors into one allocation"
/// design in 100% safe Rust. Panics if the ranges overlap or run out of bounds.
pub(crate) fn split_disjoint<'a, const N: usize>(
    buf: &'a mut [f32],
    ranges: [(usize, usize); N],
) -> [&'a mut [f32]; N] {
    for &(start, len) in &ranges {
        assert!(start + len <= buf.len(), "tensor view out of bounds");
    }
    // walk the ranges left to right, slicing each one off the remainder
    let mut order: Vec<usize> = (0..N).collect();
    order.sort_by_key(|&i| ranges[i].0);
    let mut slots: Vec<Option<&'a mut [f32]>> = (0..N).map(|_| None).collect();
    let mut rest = buf;
    let mut prev_end = 0;
    for &i in &order {
        let (start, len) = ranges[i];
        assert!(start >= prev_end, "tensor views must be pairwise disjoint");
        let (_, after) = rest.split_at_mut(start - prev_end);
        let (slice, tail) = after.split_at_mut(len);
        slots[i] = Some(slice);
        rest = tail;
        prev_end = start + len;
    }
    let mut result: [Option<&'a mut [f32]>; N] = std::array::from_fn(|_| None);
    for (i, slot) in slots.into_iter().enumerate() {
        result[i] = slot;
    }
    result.map(|slot| slot.expect("slot filled"))
}
