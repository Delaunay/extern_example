# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import time
import random


def my_dataloader_creator():
    return list(range(6000))


def my_optimizer_creator():
    return lambda : print("optimized!")


def my_criterion_creator():
    def criterion(*args, **kwargs):
        return random.normalvariate(0, 1)
    return criterion


def main():
    dataloader = my_dataloader_creator()
    criterion = my_criterion_creator()
    
    for epoch in range(10000):
        for i in dataloader:
            # avoid .item()
            # avoid torch.cuda; use accelerator from torchcompat instead
            # avoid torch.cuda.synchronize or accelerator.synchronize
            
            # y = model(i)
            # loss = criterion(y)
            # loss.backward()
            # optimizer.step()

            criterion()
            
            time.sleep(0.1)

    assert epoch < 2, "milabench stopped the train script before the end of training"
    assert i < 72, "milabench stopped the train script before the end of training"


if __name__ == "__main__":
    main()
