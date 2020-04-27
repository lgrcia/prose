import prose
from prose import pipeline

### Loading a phots file
phot = prose.Photometry("...")

# show stack
phot.show_stack()

# plot target light curve
phot.lc.plot()

# plot another star light curve (at other aperture)
phot.lcs[5].plot(aperture=4)


### Full reduction

reduction = pipeline.Reduction(folder, deepness=2)

reduction.describe()
reduction.set_observation(0)

reduction.run(destination, save_gif=gif)

photometry = pipeline.Photometry(destination)
photometry.run()

