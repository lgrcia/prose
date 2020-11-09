from prose import load

obs = load("/Users/lionelgarcia/Code/prose/test/fake_telescope_20200229_prose_I+z/fake_telescope_20200229_prose_I+z.phot")
obs.target = 1
obs = obs.broeg2005()
obs.plot_raw_diff()

g = 4