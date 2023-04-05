from prose.core.image import Image, Buffer
import numpy as np


def test_init_append(n=5):
    buffer = Buffer(n)
    init = np.random.randint(0, 20, size=buffer.mid_index + 1)
    buffer.init(init)
    np.testing.assert_allclose(buffer.buffer[buffer.mid_index : :], init)
    buffer.append(4)
    assert buffer.buffer[-1] == 4
