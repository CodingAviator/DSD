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

#To create a node
class node:
    def __init__(self,data=None,next=None):
        self.data=data
        self.next=next

#Make a link with single head
class linkedlist:
    def __init__ (self):
        self.head=None
#To insert at the beginning
    def insertatbeg (self,new.data):
        newnode=node(new.data)
        newnode.next=None
        self.head=newnode
#To insert at last
    def insertatlast (self,newdata):
        newnode=node(newdata)
        if self.head is None:
            self.head=newnode
            return
        current=self.head
        while(current.next):
            current=current.next
            newnode.next=None
            current.next=newnode


### Double Linked List
##Inserting a new node in a doubly linked list is very similar to inserting new node in linked list.
##There is a little extra work required to maintain the link of the previous node.
#Creation of node
class node:
    def __init__(self,data=None,prev=None,next=None):
        self.data=data
        self.prev=prev
        self.none=None
#Creation of link in DLL
class DoubleLL:
    def __init__(self):
        self.head=None
#To check if DLL is empty
    def is_empty(self):
        return self.head is None
#Insertion at beginning
def Inseratbeg(self,data):
    newnode=node(data,None,self.head)
    if(self.head):
        self.head.prev=newnode
        newnode.next=self.head
        self.head=newnode
#Search a node
def search(self,key):
    current=self.head
    while current:
        if current.data==key:
            return current
        current=current.next
    return None

