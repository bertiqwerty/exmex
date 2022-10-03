
    /// This tracks which numbers have already been consumed during the evaluation of a [`FlatEx`] 
    /// and are to be ignored for future operations. It is basically a vector of 
    /// bools.
    pub trait NumberTracker {
        /// Return the absolute distance to the closest unignored number in 0..=idx
        fn get_previous(&self, idx: usize) -> usize;

        /// Return the absolute distance to the next unignored number in (idx+1)..
        fn get_next(&self, idx: usize) -> usize;

        /// Mark idx as ignored
        fn ignore(&mut self, idx: usize);

        /// Return the absolute distance to the next unignored number in (idx+1).. and mark it as
        /// ignored
        #[inline(always)]
        fn consume_next(&mut self, idx: usize) -> usize {
            let next = self.get_next(idx);
            self.ignore(idx + next);
            next
        }

        /// Maximum amount of numbers that can be tracked with self
        fn max_len(&self) -> usize;
    }

    impl NumberTracker for usize {
        #[inline(always)]
        fn get_previous(&self, idx: usize) -> usize {
            let rotated = self.rotate_right(idx as u32 + 1);
            rotated.leading_ones() as usize
        }

        #[inline(always)]
        fn get_next(&self, idx: usize) -> usize {
            let rotated = self.rotate_right(idx as u32 + 1);
            rotated.trailing_ones() as usize + 1
        }

        #[inline(always)]
        fn ignore(&mut self, idx: usize) {
            *self |= 1 << idx;
        }

        fn max_len(&self) -> usize {
            Self::BITS as usize
        }
    }

    impl NumberTracker for [usize] {
        fn get_previous(&self, idx: usize) -> usize {
            let segment = idx / (usize::BITS as usize);
            let bit = idx % usize::BITS as usize;

            // use the single usize fast path which might lead to synergies with `get_next`
            let mut ones = self[segment].get_previous(bit).min(bit + 1);

            if ones == bit + 1 {
                for &word in self[..segment].iter().rev() {
                    if word == usize::MAX {
                        ones += 64;
                    } else {
                        ones += word.leading_ones() as usize;
                        break
                    }
                }
            }
            ones
        }

        fn get_next(&self, idx: usize) -> usize {
            let segment = idx / usize::BITS as usize;
            let bit = idx % usize::BITS as usize;

            // use the single usize fast path which might lead to synergies with `get_previous`
            let mut ones = self[segment].get_next(bit).min(usize::BITS as usize - bit);

            if ones == usize::BITS as usize - bit {
                for &word in self[segment..].iter().skip(1) {
                    if word == usize::MAX {
                        ones += 64;
                    } else {
                        ones += word.trailing_ones() as usize;
                        break
                    }
                }
            }
            ones
        }

        fn ignore(&mut self, idx: usize) {
            let segment = idx / usize::BITS as usize;
            let bit = idx % usize::BITS as usize;
            self[segment] |= 1 << bit;
        }

        fn max_len(&self) -> usize {
            self.len() * usize::BITS as usize
        }
    }

    #[cfg(test)]
    mod test {
        use super::NumberTracker;

        fn assert_functionality<N: NumberTracker + ?Sized>(tracker: &mut N) {
            for idx in 0..tracker.max_len() {
                assert_eq!(0, tracker.get_previous(idx));
                assert_eq!(1, tracker.get_next(idx));
            }

            let start = 23;
            let test_len = tracker.max_len() - start - 1;

            tracker.ignore(start);
            assert_eq!(1, tracker.get_previous(start));
            assert_eq!(1, tracker.get_next(start));

            for ii in 0..test_len {
                if ii % 2 == 0 {
                    assert_eq!(1, tracker.consume_next(start + ii))
                } else {
                    assert_eq!(1, tracker.get_next(start + ii));
                    tracker.ignore(start + ii + 1);
                    assert_eq!(2, tracker.get_next(start + ii));
                }
                assert_eq!(1, tracker.get_previous(start));
                assert_eq!(2 + ii, tracker.get_next(start));

                for jj in 0..(ii + 2) {
                    assert_eq!(1 + jj, tracker.get_previous(start+jj));
                    assert_eq!(2 + ii - jj, tracker.get_next(start+jj));
                }
                for jj in (ii + 2)..test_len {
                    assert_eq!(0, tracker.get_previous(start+jj));
                    assert_eq!(1, tracker.get_next(start+jj));
                }
            }
        }

        fn assert_lower_boundary_works<N: NumberTracker + ?Sized>(tracker: &mut N) {
            for idx in 1..tracker.max_len() {
                assert_eq!(0, tracker.get_previous(idx));
                assert_eq!(1, tracker.get_next(idx));
                tracker.ignore(idx);
                assert_eq!(idx, tracker.get_previous(idx));
                assert_eq!(1, tracker.get_next(idx));
            }
            assert_eq!(0, tracker.get_previous(0));
            // return value of get_next is nonsensical now
        }

        fn assert_upper_boundary_works<N: NumberTracker + ?Sized>(tracker: &mut N) {
            let last = tracker.max_len() - 1;
            for distance in 0..last {
                let idx = last - distance;
                assert_eq!(0, tracker.get_previous(idx));
                assert_eq!(1 + distance, tracker.get_next(idx));
                tracker.ignore(idx);
                assert_eq!(1, tracker.get_previous(idx));
                assert_eq!(1 + distance, tracker.get_next(idx));
            }
            assert_eq!(last, tracker.get_previous(last));
            // return value of get_next(last) is nonsensical now
        }



        #[test]
        fn test_scalar() {
            assert_functionality(&mut 0);
            assert_lower_boundary_works(&mut 0);
            assert_upper_boundary_works(&mut 0);
        }

        #[test]
        fn test_slice() {
            assert_functionality([0, 0, 0, 0].as_mut_slice());
            assert_lower_boundary_works([0, 0, 0, 0].as_mut_slice());
            assert_upper_boundary_works([0, 0, 0, 0].as_mut_slice());
        }
    }