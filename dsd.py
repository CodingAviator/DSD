##### UNIT-2

### Stack ADT
##A stack is a linear data structure that stores items in a Last-In/First-Out (LIFO) or First-In/Last-Out (FILO) manner. 
##In stack, a new element is added at one end and an element is removed from that end only. 
##The insert and delete operations are often called push and pop.
##empty() – Returns whether the stack is empty – Time Complexity: O(1)
##size() – Returns the size of the stack – Time Complexity: O(1)
##top() / peek() – Returns a reference to the topmost element of the stack – Time Complexity: O(1)
##push(a) – Inserts the element ‘a’ at the top of the stack – Time Complexity: O(1)
##pop() – Deletes the topmost element of the stack – Time Complexity: O(1)
class Stack:
    def __init__(self):
        self.items=[]
    def isempty(self):
        return self.items==[]
    def push(self,item):
        self.items.append(item)
        top=top+1
    def pop(self):
        return self.items.pop()
    def size(self):
        return len(self.items)
s=Stack()
s.push("Stack")
s.push("True")
print(s.size())
print(s.isempty())
print(s.pop())
print(s.size())




### Queue ADT
##the queue is a linear data structure that stores items in a First In First Out (FIFO) manner. 
##With a queue, the least recently added item is removed first. 
##A good example of a queue is any queue of consumers for a resource where the consumer that came first is served first.
##Enqueue: Adds an item to the queue. If the queue is full, then it is said to be an Overflow condition – Time Complexity : O(1)
##Dequeue: Removes an item from the queue. The items are popped in the same order in which they are pushed. If the queue is empty, then it is said to be an Underflow condition – Time Complexity : O(1)
##Front: Get/remove the front item from queue – Time Complexity : O(1)
##Rear: Get/insert/enqueue the last item from queue – Time Complexity : O(1)
class Queue:
    def __init__(self,capacity):
        self.capacity=capacity
        self.queue=[None]
        self.front=self.rear=-1
    def isempty(self):
        return self.front==-1
    def isfull(self):
        self.capacity=self.front
        return (self.rear+1)%1
    def enqueue(self):
        if self.isfull():
            print("Queue is full")
            return None
        if self.isempty():
            self.front=()
            self.rear=(self.rear+1)%self.capacity
            self.queue=(self.rear)=self.item
    def dequeue(self):
        if self.isempty():
            print("Queue is empty")
            return None
        item=self.queue[self.front]
        if self.front==self.rear:
            self.front=self.rear=-1
        else:
            self.front=(self.front+1)//self.capacity
Q=Queue(capacity=5)
Q.enqueue(5)
Q.enqueue("True")
Q.enqueue("ADT")
Q.dequeue()




### Single Linked List
##In a singly linked list, each node consists of two parts: data and a pointer to the next node. 
##The data part stores the actual information, while the pointer (or reference) part stores the address of the next node in the sequence. 
##This structure allows nodes to be dynamically linked together, forming a chain-like sequence.

# create a node
class Node:
  # constructor
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
# Creating a single node
first = Node(3)
print(first.data)

# create a linked list
class Node:
    def __init__(self, data = None, next=None):
        self.data = data
        self.next = next
# A Linked List class with a single head node
class LinkedList:
    def __init__(self):
        self.head = None
    # Insert at the beginning
    def insertAtBeginning(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node
    # Insert after a node
    def insertAfter(self, key, new_data):
        prev_node = self.search(key)
        if prev_node is None:
            print("The given previous node or search node must be in the LinkedList.")
            return
        new_node = Node(new_data)
        new_node.next = prev_node.next
        prev_node.next = new_node
    # Search an element
    def search(self, key):
        current = self.head
        while current:
            if current.data == key:
                return current
            current = current.next
        return None
    #insert @ the last or tail
    def insertAtEnd(self, new_data):
        new_node = Node(new_data)
        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while (last.next):
            last = last.next
        new_node.next=None
        last.next = new_node
    #delete a node @ anywhere
    def deletenode(self, key):
        while(self.head != None and self.head.data == key):
            nodeToDelete = self.head
            self.head = self.head.next
            nodeToDelete = None
        temp = self.head
        if(temp != None):
            while(temp.next != None):
                if(temp.next.data == key):
                    nodeToDelete = temp.next
                    temp.next = temp.next.next
                    nodeToDelete = None
                else:
                    temp = temp.next
    # Print the linked list
    def printList(self):
        temp = self.head
        while (temp):
            print(str(temp.data) + " ", end="")
            temp = temp.next
        print()
# Linked List with a single node
LL = LinkedList()
LL.head = Node(3)
print(LL.head.data)
LL.insertAtBeginning(2)
LL.printList()
LL.insertAfter(2,5)
LL.printList()
LL.insertAtEnd(9)
LL.printList()
LL.deletenode(2)
LL.printList()




### Double Linked List
##Inserting a new node in a doubly linked list is very similar to inserting new node in linked list.
##There is a little extra work required to maintain the link of the previous node.
#Creation of node
class Node:
    def __init__(self, data=None, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next
class DoublyLinkedList:
    def __init__(self):
        self.head = None
    def is_empty(self):
        return self.head is None
    def insert_at_beginning(self, data):
        new_node = Node(data, None, self.head)
        if self.head:
            self.head.prev = new_node
            new_node.next=self.head
            self.head = new_node
        self.head = new_node
    def insert_after(self, key, data):
        prev_node = self.search(key)
        if prev_node is None:
            print("The given search (key) node must be in the LinkedList.")
            return
        new_node = Node(data, prev_node, prev_node.next)
        if prev_node.next:
          new_node.next=prev_node.next
          prev_node.next= new_node
          new_node.prev=prev_node
          new_node.next.prev=new_node
        else:
          new_node.next=prev_node.next
          prev_node.next = new_node
          new_node.prev=self.head
    def insert_at_end(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            last = self.head
            while last.next:
                last = last.next
            new_node.prev = last
            last.next = new_node
    def delete_node(self, key):
        current = self.head
        while current:
            if current.data == key:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                current = None
                return
            current = current.next
    def search(self, key):
        current = self.head
        while current:
            if current.data == key:
                return current
            current = current.next
        return None
    def print_list(self):
        temp = self.head
        while temp:
            print(str(temp.data) + " ", end="")
            temp = temp.next
        print()
doubly_linked_list = DoublyLinkedList()
doubly_linked_list.head = Node(3)
print(doubly_linked_list.head.data)
doubly_linked_list.insert_at_beginning(2)
doubly_linked_list.print_list()
doubly_linked_list.insert_after(2, 5)
doubly_linked_list.print_list()
doubly_linked_list.insert_at_end(9)
doubly_linked_list.print_list()
doubly_linked_list.delete_node(2)
doubly_linked_list.print_list()
doubly_linked_list.delete_node(3)
doubly_linked_list.print_list()




# infix to postfix conversion
class InfixToPostfixConverter:
    def __init__(self):
        self.stack = []
        self.output = []
    def is_operator(self, char):
        return char in {'+', '-', '*', '/'}
    def precedence(self, operator):
        if operator == '+' or operator == '-':
            return 1
        elif operator == '*' or operator == '/':
            return 2
        return 0
    def infix_to_postfix(self, infix_expression):
        for char in infix_expression:
            if char.isalnum():
                self.output.append(char)
            elif char == '(':
                self.stack.append(char)
            elif char == ')':
                while self.stack and self.stack[-1] != '(':
                    self.output.append(self.stack.pop())
                self.stack.pop()  # Discard '('
            elif self.is_operator(char):
                while (self.stack and
                       self.precedence(char) <= self.precedence(self.stack[-1])):
                    self.output.append(self.stack.pop())
                self.stack.append(char)
        while self.stack:
            self.output.append(self.stack.pop())
        return ''.join(self.output)
infix_expression = "a+b*(c+d)/e"
converter = InfixToPostfixConverter()
postfix_expression = converter.infix_to_postfix(infix_expression)
print(f"Infix Expression: {infix_expression}")
print(f"Postfix Expression: {postfix_expression}")


# evaluation of postfix expression
class PostfixEvaluator:
    def __init__(self):
        self.stack = []
    def evaluate_postfix(self, postfix_expression):
        for char in postfix_expression:
            if char.isdigit():
                self.stack.append(int(char))
            elif char == ' ':
                continue
            else:
                operand2 = self.stack.pop()
                operand1 = self.stack.pop()
                result = self.perform_operation(operand1, operand2, char)
                self.stack.append(result)
        return self.stack.pop()
    def perform_operation(self, operand1, operand2, operator):
        if operator == '+':
            return operand1 + operand2
        elif operator == '-':
            return operand1 - operand2
        elif operator == '*':
            return operand1 * operand2
        elif operator == '/':
            return operand1 / operand2
postfix_expression = "23*5+"
evaluator = PostfixEvaluator()
result = evaluator.evaluate_postfix(postfix_expression)
print(f"Postfix Expression: {postfix_expression}")
print(f"Result: {result}")


#balancing of postfix expression
class ParenthesisBalancer:
    def __init__(self):
        self.stack = []
    def is_opening_parenthesis(self, char):
        return char in {'(', '[', '{'}
    def is_closing_parenthesis(self, char):
        return char in {')', ']', '}'}
    def is_matching_pair(self, open_parenthesis, close_parenthesis):
        pairs = {'(': ')', '[': ']', '{': '}'}
        return pairs[open_parenthesis] == close_parenthesis
    def check_balancing(self, expression):
        for char in expression:
            if self.is_opening_parenthesis(char):
                self.stack.append(char)
            elif self.is_closing_parenthesis(char):
                if not self.stack or not self.is_matching_pair(self.stack.pop(), char):
                    return False
        return not self.stack
expression = "("
balancer = ParenthesisBalancer()
if balancer.check_balancing(expression):
  print(f"The expression {expression} is balanced.")
else:
  print(f"The expression {expression} is not balanced.")


def check_balancing(expression):
    stack = []
    for char in expression:
        if char in {'(', '[', '{'}:
            stack.append(char)
        elif char in {')', ']', '}'}:
            if not stack or not is_matching_pair(stack.pop(), char):
                return False
    return not stack
def is_matching_pair(opening, closing):
    return (opening == '(' and closing == ')') or \
           (opening == '[' and closing == ']') or \
           (opening == '{' and closing == '}')
# Example usage
expression = "{[()]})"
if check_balancing(expression):
    print(f"{expression} is balanced.")
else:
    print(f"{expression} is not balanced.")




##### UNIT-3
### Linear Search
class LinearSearch:
    def __init__(self, data):
        self.data = data
    def search(self, target):
        for i, value in enumerate(self.data):
            if value == target:
                return i  # Return the index if the target is found
        return -1  # Return -1 if the target is not found
    
# Create an instance of the LinearSearch class
my_list = [64, 34, 25, 12, 22, 11, 90]
linearobj = LinearSearch(my_list)
# Define the target element to search for
target_element = 22
# Perform linear search
result = linearobj.search(target_element)
# Print the result
if result != -1:
    print(f"Element {target_element} found at index {result}.")
else:
    print(f"Element {target_element} not found.")



### Binary Search
class BinarySearch:
    def __init__(self, array):
        self.array = array
    def search(self, x):
        low, high = 0, len(self.array) - 1
        return self.binary_search(x, low, high)
    def binary_search(self, x, low, high):
        if high >= low:
            mid = low + (high - low) // 2
            if self.array[mid] == x:
                return mid
            elif self.array[mid] > x:
                return self.binary_search(x, low, mid - 1)
            else:
                return self.binary_search(x, mid + 1, high)
        else:
            return -1
# Example usage:
array = [3, 4, 5, 6, 7, 8, 9]
x = 9
binary_search_instance = BinarySearch(array)
result = binary_search_instance.search(x)
if result != -1:
    print("Element  is present at index " + str(result))
else:
    print("Not found")




### Bubble Sort
class Bubblesort:
    def __init__ (self,data):
        self.data=data
        
    def sort(self):
        n=len(self.data)
        for i in range(n):
            for j in range(0,n-i-1):
                if self.data[j] > self.data[j + 1]:
                    self.data[j],self.data[j+1]=self.data[j+1],self.data[j]

data=[-2,45,0,11,-9]
BS=Bubblesort(data)
BS.sort()
print("sorted array in ascending order")
print(data)



### Selection Sort
class  Selectionsort:
    def __init__ (self,data):
        self.data=data

    def sort(self):
        n=len(self.data)
        for i in range (n-1):
            min1=i
            for j in range(i+1,n):
                if self.data[j]<self.data[min1]:
                    min1=j

            self.data[i],self.data[min1]=self.data[min1],self.data[i]

    def display(self):
        print("Sorted array:",self.data)

arr=[13,25,28,42,63]
sorter=Selectionsort(arr)
sorter.sort()
sorter.display()



### Insertion Sort
class Insertion:
    def __init__(self,arr):
        self.arr=arr

    def sort(self):
        for i in range(1, len(self.arr)):
            key = self.arr[i]
            j = i - 1
            while j >= 0 and key < self.arr[j]:
                self.arr[j + 1] = self.arr[j]
                j = j-1
            self.arr[j + 1] = key

        self.display()

    def display(self):
        print("Sorted Array:")
        print(self.arr)

arr=[2,5,1,8,3,56,74,33]
obj=Insertion(arr)
obj.sort()



### Merge Sort
class Merge:
    def __init__(self,arr):
        self.arr=arr

    def sort(self):
        if len(self.arr) > 1:
            mid = len(self.arr)//2
            left_half = self.arr[:mid]
            right_half = self.arr[mid:]
            left_merge_sort = Merge(left_half)
            right_merge_sort = Merge(right_half)
            left_merge_sort.sort()
            right_merge_sort.sort()

            i = j = k = 0

            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    self.arr[k] = left_half[i]
                    i += 1
                else:
                    self.arr[k] = right_half[j]
                    j += 1
                k += 1
            while i < len(left_half):
                self.arr[k] = left_half[i]
                i += 1
                k += 1
            while j < len(right_half):
                self.arr[k] = right_half[j]
                j += 1
                k += 1

    def display(self):
        print("Sorted Array:")
        print(self.arr)

arr=[11,23,45,63,22,1,42]
obj=Merge(arr)
obj.sort()
obj.display()



### Quick Sort
class Quick:
    def __init__(self,arr):
        self.arr=arr

    def sort(self):
        self.quickSort(0, len(self.arr) - 1)

    def quickSort(self,first,last):
        if first < last:
            partitionindex = self.partition(first, last)
            self.quickSort(first, partitionindex - 1)
            self.quickSort(partitionindex + 1, last)

    def partition(self, first, last):
        pivotvalue = self.arr[first]
        leftmark = first + 1
        rightmark = last
        done = False

        while not done:
            while leftmark <= rightmark and self.arr[leftmark] <=pivotvalue:
                leftmark = leftmark + 1

            while self.arr[rightmark] >= pivotvalue and rightmark >=leftmark:
                rightmark = rightmark - 1

            if rightmark < leftmark:
                done = True
            else:
                self.arr[leftmark],self.arr[rightmark] = self.arr[rightmark],self.arr[leftmark]

        self.arr[first],self.arr[rightmark]=self.arr[rightmark],self.arr[first]
        return rightmark

    def display(self):
        print("Sorted Array:")
        print(self.arr)

arr=[12,1,56,3,7,33]
obj=Quick(arr)
obj.sort()
obj.display()





##### UNIT-4

### Implementation Of Tree Traversal
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key 
def printInorder(root):
    if root:
        printInorder(root.left)
        print(root.val),
        printInorder(root.right) 
def printPostorder(root):
    if root:
        printPostorder(root.left)
        printPostorder(root.right)
        print(root.val), 
def printPreorder(root):
    if root:
        print(root.val),
        printPreorder(root.left)
        printPreorder(root.right)
root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5) 
print ("Preorder traversal of binary tree is") 
printPreorder(root) 
print ("\nInorder traversal of binary tree is") 
printInorder(root) 
print ("\nPostorder traversal of binary tree is") 
printPostorder(root)



### Implemmentation Of Binary Search Tree
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
def inorder(root):
    if root is not None:
        inorder(root.left)
        print(str(root.key) + "->", end=' ')
        inorder(root.right)
def insert(node, key):
    if node is None:
         return Node(key)       
    if key < node.key:
        node.left = insert(node.left, key)
    else:
        node.right = insert(node.right, key)
    return node
def minValueNode(node):
    current = node
    while(current.left is not None):
        current = current.left
    return current
def deleteNode(root, key):
    if root is None:
        return root
    if key < root.key:
        root.left = deleteNode(root.left, key)
    elif(key > root.key):
        root.right = deleteNode(root.right, key)
    else:
        if root.left is None:
            temp = root.right
            root = None
            return temp
        elif root.right is None:
            temp = root.left
            root = None
            return temp
        temp = minValueNode(root.right)
        root.key = temp.key
        root.right = deleteNode(root.right, temp.key)
        return root
root = None
root = insert(root, 8)
root = insert(root, 3)
root = insert(root, 1)
root = insert(root, 6)
root = insert(root, 7)
root = insert(root, 10)
root = insert(root, 14)
root = insert(root, 4)
print("Inorder traversal: ", end=' ')
inorder(root)
print("\nDelete 10")
root = deleteNode(root, 10)
print("Inorder traversal: ", end=' ')
inorder(root)



### Graph Representation And Traversal Algorithm(DFS)
from collections import defaultdict
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    def addEdge(self, u, v):
        self.graph[u].append(v)
    def DFSUtil(self, v, visited):
        visited.add(v)
        print(v, end=' ')
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)
    def DFS(self, v):
        visited = set()
        self.DFSUtil(v, visited)
g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
print("Following is DFS from (starting from vertex 2)")
g.DFS(2)




##### UNIT-5
### SINGLE SOURCE SHORTEST PATH ALGORITHM
from numpy import Inf
class DijkstraAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = len(graph)
        self.distances = [Inf for _ in range(self.num_nodes)]
        self.visited = [False for _ in range(self.num_nodes)]
    def run_dijkstra(self, start):
        self.distances[start] = 0
        for _ in range(self.num_nodes):
            u = self._get_min_distance_node()
            if self.distances[u] == Inf:
                break
            self.visited[u] = True
            for v, d in self.graph[u]:
                if self.distances[u] +d< self.distances[v]:
                    self.distances[v] = self.distances[u] + d
        return self.distances
    def _get_min_distance_node(self):
        u = -1
        for x in range(self.num_nodes):
            if not self.visited[x] and (u == -1 or self.distances[x]< self.distances[u]):
                u = x
        return u
graph = {
0: [(1, 1)],
1: [(0, 1), (2, 2), (3, 3)],
2: [(1, 2), (3, 1), (4, 5)],
3: [(1, 3), (2, 1), (4, 1)],
4: [(2, 5), (3, 1)]
}
dijkstra_instance = DijkstraAlgorithm(graph)
result_distances = dijkstra_instance.run_dijkstra(0)
print(result_distances)



### MINIMUM SPANNING TREE IMPLEMENTATION (KRUSKAL)
class Vertex:
    def __init__(self, key):
        self.key = key
        self.parent = self
        self.rank = 0
class Graph:
    def __init__(self):
        self.vertices={}
        self.edges=[]
    def add_vertex(self, key):
        self.vertices[key] = Vertex(key)
    def add_edge(self, start, end, weight):
        self.edges.append((start, end, weight))
    def find(self, vertex):
        if vertex != vertex.parent:
            vertex.parent = self.find(vertex.parent)
        return vertex.parent
    def union(self, root1, root2):
        if root1.rank>root2.rank:
            root2.parent = root1
        elif root1.rank < root2.rank:
            root1.parent = root2
        else:
            root2.parent = root1
            root1.rank += 1
    def mst_kruskal(self):
        mst = Graph()
        self.edges.sort(key=lambda x: x[2])
        for vertex_key in self.vertices:
            mst.add_vertex(vertex_key)
        for edge in self.edges:
            start, end, weight = edge
            root1 = self.find(self.vertices[start])
            root2 = self.find(self.vertices[end])
            if root1 != root2:
                mst.add_edge(start, end, weight)
                self.union(root1, root2)
        return mst
g = Graph()
g.add_vertex('A')
g.add_vertex('B')
g.add_vertex('C')
g.add_vertex('D')
g.add_edge('A','B',1)
g.add_edge('A','C',4)
g.add_edge('B','C',2)
g.add_edge('B','D',5)
g.add_edge('C','D',1)
mst = g.mst_kruskal()
print("Minimum Spanning Tree:")
for edge in mst.edges:
    print(edge)



### MINIMUM SPANNING TREE IMPLEMENTATION (PRIMS ALGORITHM)
import sys
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in
                      range(vertices)]
    def add_edge(self, u, v, weight):
        self.graph[u][v] = weight
        self.graph[v][u] = weight
    def print_mst(self, parent):
        print("Edge \tWeight:")
        for i in range(1, self.V):
            print(f"{parent[i]} - {i}\t{self.graph[i][parent[i]]}")
    def min_key(self, key, mst_set):
        min_value = sys.maxsize
        min_index = -1
        for v in range(self.V):
            if key[v]<min_value and not mst_set[v]:
                min_value = key[v]
                min_index = v
        return min_index
    def prim_mst(self):
        key = [sys.maxsize] * self.V
        parent = [-1] * self.V
        key[0] = 0
        mst_set = [False] * self.V
        for _ in range(self.V):
            u = self.min_key(key, mst_set)
            mst_set[u] = True
            for v in range(self.V):
                if 0<self.graph[u][v]<key[v] and not mst_set[v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        self.print_mst(parent)
g = Graph(5)
g.add_edge(0, 1, 2)
g.add_edge(0, 3, 6)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 8)
g.add_edge(1, 4, 5)
g.add_edge(2, 4, 7)
g.add_edge(3, 4, 9)
g.prim_mst()
