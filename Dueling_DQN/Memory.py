import numpy as np
from collections import deque

MEMORY_LEN = 2000
  
class Memory:
  """
  Save memory and pick random batch
  """
  def __init__(self, names, mem_len=MEMORY_LEN):
    """
    :param names: names of memroy slices
    :param mem_len: length of memory
    """
    assert isinstance(names, list)
    assert isinstance(mem_len, int) and mem_len > 0

    self.names = names
    self.meomries = {}
    for name in self.names:
      assert isinstance(name, str)
      self.meomries[name] = deque(maxlen=mem_len)

  def remember(self, memory):
    """
    insert a slice of memory
    
    :param memory: memroy slice
    """
    assert isinstance(memory, dict)

    for name in self.names:
      assert name in memory
      self.meomries[name].append(memory[name])

  def random_batch(self, size):
    """
    pick random batch of memories

    :param size: size of batch
    """
    assert isinstance(size, int) and size > 0

    mem_len = len(self.meomries[self.names[0]])
    size = min(size, mem_len)
    indexes = np.random.choice(mem_len, size)

    batch = {}
    for name in self.names:
      batch[name] = np.array(self.meomries[name])[indexes]

    return batch

