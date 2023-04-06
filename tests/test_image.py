from prose.core.image import Image, Buffer
import numpy as np


def test_init_append(n=5):
    buffer = Buffer(n)
    init = np.random.randint(0, 20, size=buffer.mid_index + 1)
    buffer.init(init)
    current = next(iter(buffer))
    np.testing.assert_equal(current, init[0])
    np.testing.assert_allclose(buffer.buffer[buffer.mid_index : :], init)
    buffer.append(4)
    assert buffer.buffer[-1] == 4

def test_buffer_iter():
    buffer = Buffer(5)
    datas = np.random.randint(0, 20, 20)
    buffer.init(datas)
    for i, current in enumerate(buffer):
        print(buffer.buffer)
        assert current == datas[i]