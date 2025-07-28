import heapq
import numpy as np

class BoundedPriorityQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []  # Min-heap storing (-priority, vector)
        self.vector_keys = set()  # For O(1) duplicate checks
        self._counter = 0

    def _vector_key(self, vector: np.ndarray):
        return hash(vector.tobytes())  # Unique key for each vector

    def initialize(self, vectors, priorities):
        assert len(vectors) == len(priorities)
        # Convert to internal representation: (-priority, vector)
        items = [(-p, i, v) for i, (p, v) in enumerate(zip(priorities, vectors))]
        self._counter = len(items)

        if len(items) <= self.max_size:
            self.heap = items
            heapq.heapify(self.heap)
        else:
            # Use heap to select top max_size smallest priorities
            heap = []
            for item in items:
                if len(heap) < self.max_size:
                    heapq.heappush(heap, item)
                else:
                    if item > heap[0]:  # item[0] is -priority (larger = better)
                        heapq.heapreplace(heap, item)
            self.heap = heap

        self.vector_keys = {self._vector_key(v) for _, _, v in self.heap}

    def add_batch(self, vectors, priorities):
        assert len(vectors) == len(priorities)
        saved = 0
        
        for vector, priority in zip(vectors, priorities):
            key = self._vector_key(vector)
            if key in self.vector_keys:
                continue  # Skip duplicate
            
            # Internal representation uses negative priority
            neg_priority = -priority
            item = (neg_priority, self._counter, vector)
            self._counter += 1

            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, item)
                self.vector_keys.add(key)
                saved += 1
            else:
                # Compare with current worst (heap[0])
                if neg_priority > self.heap[0][0]:
                    # Replace worst item
                    popped_item = heapq.heapreplace(self.heap, item)
                    self.vector_keys.remove(self._vector_key(popped_item[2]))
                    self.vector_keys.add(key)
                    saved += 1
        return saved

    def get_saved_vectors(self):
        return np.array([item[2] for item in self.heap])
    
    def get_priorities(self):
        return np.array([-item[0] for item in self.heap])

    def to_state_dict(self):
        return {
            "max_size": self.max_size,
            "heap": self.heap,
            "vector_keys": list(self.vector_keys),
            "counter": self._counter
        }
    
    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state["max_size"])
        obj.heap = state["heap"]
        heapq.heapify(obj.heap)  # Maintain heap property
        obj.vector_keys = set(state["vector_keys"])
        obj._counter = state["counter"]
        return obj
    
    def __len__(self):
        return len(self.heap)