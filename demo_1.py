import numpy as np
from tensorboardX import SummaryWriter


writer = SummaryWriter()
for epoch in range(100):
	writer.add_scalar('scalar/test', np.random.rand(), epoch)
	writer.add_scalars('scalar/scalars_test', {"xsinx":epoch * np.sin(epoch), "xcosx":epoch * np.cos(epoch)}, epoch)

writer.close()

# tensorboard --logdir runs  

