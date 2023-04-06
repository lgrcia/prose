from prose.core.image import Image, Buffer
import numpy as np


def test_init_append(n=5):
    buffer = Buffer(n)
    init = np.random.randint(0, 20, size=20)
    buffer.init(init)
    np.testing.assert_equal(
        buffer.items[buffer.mid_index + 1 :], init[: buffer.mid_index]
    )
    buffer.append(4)
    assert buffer.items[-1] == 4


def test_buffer_iter():
    buffer = Buffer(5)
    data = np.random.randint(0, 20, 20)
    buffer.init(data)
    for i, buf in enumerate(buffer):
        assert buf.current == data[i]
