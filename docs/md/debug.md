# Debugging

Finding why a specific error is thrown when running a [Sequence](prose.Sequence) can be challenging. Here are a few steps to debug a [Sequence](prose.Sequence).

## 1. Find from which block the error comes from

The error might be on the [Sequence](prose.Sequence) `_run` function, but scrolling up will reveal in which block the error actually occurs (if not specified). For each block, the documentation (should) contain explanation for each possible exceptions being raised. If not, [open a Github issue](https://github.com/lgrcia/prose/issues/new/choose)!

## 2. Show the last image

A [Sequence](prose.Sequence) holds a `last_image` attribute that corresponds to the image being processed by the blocks. When an error occurs it might be helpful to show this last image with

```python
sequence.last_image.show()
```

to realize why the next block struggles (such as because sources not being well detected in the image)

## 3. Run all blocks individually

If None of this methods help, you can always load an image by hand and run all blocks manually. That's what a [Sequence](prose.Sequence) does internally. Here is how to do it

```python
from prose import FITSImage

# your test image
test_image = FITSImage(your_path)

# your sequence
sequence = Sequence([...])

# run all blocks
for block in sequence.blocks:
    test_image = block(test_image)

# terminate all blocks
for block in sequence.blocks:
    block.terminate()

```

This way, the error will be thrown on a specific block, and you can track the changes of each of them on your test image

And if you have any question, just [open a Github issue](https://github.com/lgrcia/prose/issues/new/choose)!