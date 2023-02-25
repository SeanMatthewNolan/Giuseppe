from typing import Tuple


def make_array_slices(_component_lengths: Tuple[int, ...]) -> Tuple[slice, ...]:
    slice_list = [slice(_component_lengths[0])]
    for n in _component_lengths[1:]:
        prev_slice_stop = slice_list[-1].stop
        slice_list.append(slice(prev_slice_stop, prev_slice_stop + n))

    return tuple(slice_list)
