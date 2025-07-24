# -- coding: utf-8 --
from typing import Union, Iterable


class Node:
    def __init__(self, val, prev, next):
        self.data = val
        self.prev: Union[Node, None] = prev
        self.next: Union[Node, None] = next

class LinkedList:
    def __init__(self, elems: Union[list, None]):
        self.head: Union[Node, None] = None
        self.tail: Union[Node, None] = None
        self.size = 0

        if elems:
            for e in elems:
                self.insert_tail(Node(e, None, None))

    def __len__(self):
        return self.size

    def __str__(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " -> ".join(elements) + " -> None"

    def insert_tail(self, node: Node):
        if self.size == 0:
            self.head = node
            self.tail = node
            node.prev = None
            node.next = None
        else:
            self.tail.next = node
            node.prev = self.tail
            node.next = None
            self.tail = node
        self.size += 1

    def insert_head(self, node: Node):
        if self.size == 0:
            self.head = node
            self.tail = node
            node.prev = None
            node.next = None
        else:
            node.next = self.head
            node.prev = None
            self.head.prev = node
            self.head = node


if __name__ == '__main__':
    l1 = [1]
    l2 = [2]
    l3 = [3]
    l4 = [4]
    l5 = [5]
    ll = [l1, l2, l3, l4, l5]

    for l in ll:
        if l == l4:
            ll.remove(l4)
    print(ll)