# 9th program
'''
def is_Safe(board, row, col, n):
    for i in range(row):
        if board[i][col] == 1:
            return False
    
    i, j = row, col
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1
    
    i, j = row, col
    while i >= 0 and j < n:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1
    
    return True

def solve_n_queens_util(board,row,n,solutions):
    if row==n:
        solutions.append([row[:] for row in board])
        return
    for col in range(n):
        if is_Safe(board,row,col,n):
            board[row][col]=1
            solve_n_queens_util(board,row+1,n,solutions)
            board[row][col]=10

def solve_n_queens(n):
    board=[[0 for _ in range(n)] for _ in range(n)]
    solutions=[]
    solve_n_queens_util(board,0,n,solutions)
    return solutions

def print_solution(solutions):
    if not solutions:
        print("no sol found\n")
    else:
        for sol in solutions:
            for row in sol:
                print(' '.join('Q' if cell == 1 else '.' for cell in row))
 
            print()

n=int(input("enter the val for n"))
queens_sol=solve_n_queens(n)
print_solution(queens_sol)

#8th program

slots=['9','11','1','3']
subjects=['A','B','C','D','E']
enrolled_in_com_subj={}
enrolled_in_com_subj['A']=['B','C','D']
enrolled_in_com_subj['B']=['A','C','E']
enrolled_in_com_subj['C']=['A','B','D','E']
enrolled_in_com_subj['D']=['A','C','E']
enrolled_in_com_subj['E']=['B','C','D']
schedule_time_slots={}

def promising(subject,slot):
    for students_enrolled in enrolled_in_com_subj.get(subject):
        students_allot_time=schedule_time_slots.get(students_enrolled)
        if students_allot_time==slot:
            return False
    return True
    

def get_time(subject):
    for slot in slots:
        if promising(subject,slot):
            return slot
        
def main():
    for subject in subjects:
        schedule_time_slots[subject]=get_time(subject)
        print(schedule_time_slots)
main()


#Program 1 incomplete

import time
import random

class TVChannel:
    def __init__(self,name,viewers,viewing_time):
        self.name=name
        self.viewers=viewers
        self.viewing_time=viewing_time
    
def createchannel(i):
    name=f"channel{i}"
    viewers=random.randint(500,3000000)
    viewing_time=random.randint(50,180)
    return TVChannel(name,viewers,viewing_time)

def ratingchannel(channel):
    if channel.viewers>=1500000:
        return 'RANK 1'
    elif channel.viewers>=1000000:
        return 'RANK 2'
    else:
        return 'SHUT DOWN'
    
def plot_sorting_algo():
    sortingalgo=['merge_sort','bubble_sort','quick_sort','selection_sort']
    runningtimes=[]
    for algo in sortingalgo:
        data=[random.randint(1,1000) for _ in range(1000)]
        start=time.time()
        if algo=='merge_sort':
            merge_sort(data)
        elif algo=='quick_sort':
            quick_sort(data)
        elif algo=='bubble_sort':
            bubble_sort(data)
        elif algo=='selection_sort':
            selection_sort(data)
        end=time.time()
        running=end-start
        runningtimes.append(running)
    plt.bar(sortingalgo,runningtimes)
    plt.xlabel("sorting algorithms")
    plt.ylabel("running times")
    plt.title("sorting algorithm ka runing time")
    plt.show

def bubble_sort(arr):
    n=len(arr)
    for i in range(0,n-1):
        for j in range(0,n-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]

def selection_sort(arr):
    n=len(arr)
    for i in range(0,n):
        minind=i
        for j in range(i+1,n):
            if arr[j]<arr[minind]:
                minind=j
        arr[i],arr[minind]=arr[minind],arr[i]

def quick_sort(arr):
    if len(arr)<=1:
        return arr
    else:
        pivot=arr[0]
        smaller=[x for x in arr[1:] if x<=pivot]
        larger=[x for x in arr[1:] if x>pivot]
        return quick_sort(smaller)+[pivot]+quick_sort(larger)
    
def merge_sort(arr):
    if len(arr)<=1:
        return arr
    else:
        mid=(len(arr))//2
        lefthalf=arr[:mid]
        righthalf=arr[mid:]
        lefthalf=merge_sort(lefthalf)
        righthalf=merge_sort(righthalf)
        merge(lefthalf,righthalf)

def merge(left,right):
    result=[]
    i,j=0
    while i<len(left) and j<len(right):
        if left[i]<=right[j]:
            result.append(left[i])
            i+=1
        else:
            result.append(right[j])
            j+=1

    while i<len(left):
        result.append(left[i])
        i+=1
    while j<len(right):
        result.append(right[j])
        j+=1
    return result

def appmenu():
    channels=[]
    while True:
        print("enter 10 chsnnels\n")
        print("rating off chn\n")
        print("plot\n")
        print("exit\n")
        choice=input("enter ur choice:\n")
        if choice=="1":
            for i in range(0,10):
                channel=createchannel(i+1)
                channels.append(channel)
            print("created\n")
        elif choice=="2":
            for channel in channels:
                rate=ratingchannel(channel)
                print(f"{channel.name},{channel.viewers}viwers,{channel.viewing_time}hrs,{rate}")
appmenu()

#2nd program

class Job:
 def __init__(self, taskId, deadline, profit):
    self.taskId = taskId
    self.deadline = deadline
    self.profit = profit
 
def scheduleJobs(jobs, T):
    profit = 0
    slot = [-1] * T
    jobs.sort(key=lambda x: x.profit, reverse=True)
    for job in jobs:
        for j in reversed(range(job.deadline)):
                if j < T and slot[j] == -1:
                    slot[j] = job.taskId
                    profit += job.profit
                    break
    print('The scheduled jobs are:', list(filter(lambda x: x != -1, slot)))
    print('The total profit earned is:', profit)
 
if __name__ == '__main__':
    jobs = []
    n = int(input('Enter the number of jobs: '))
    for i in range(n):
        taskId = int(input(f'Enter the task ID for job {i+1}: '))
        deadline = int(input(f'Enter the deadline for job {i+1}: '))
        profit = int(input(f'Enter the profit for job {i+1}: '))
        jobs.append(Job(taskId, deadline, profit))
    T = int(input('Enter the deadline limit: '))
    scheduleJobs(jobs, T)

#3rd Program

def dij(n, v, cost):
    flag = [0 for i in range(1, n + 2, 1)]
    dist = [float('inf') for i in range(1, n + 2, 1)]
    dist[v] = 0
    path = [[] for i in range(1, n + 2, 1)]
    path[v].append(v)

    for _ in range(1, n + 1, 1):
        u = -1
        min_dist = float('inf')
        for i in range(1, n + 1, 1):
            if not flag[i] and dist[i] < min_dist:
                u = i
                min_dist = dist[i]
        if u == -1:
            break
        flag[u] = 1

        for i in range(1, n + 1, 1):
            if dist[u] + cost[u][i] < dist[i]:
                dist[i] = dist[u] + cost[u][i]
                path[i] = path[u] + [i]
    return dist, path
n = int(input("Enter the number of vertices: "))
print("Enter the weighted matrix (separated by spaces):")
cost = [[0 for j in range(1, n + 2, 1)] for i in range(1, n + 2, 1)]
for i in range(1, n + 1, 1):
    row = list(map(int, input().split()))
    for j in range(1, n + 1, 1):
        cost[i][j] = row[j - 1]
        if cost[i][j] == 0:
            cost[i][j] = float('inf')
v = int(input("Enter the source vertex: "))
dist, path = dij(n, v, cost)
print("The shortest path from source", v, "to the remaining vertices is:")
for i in range(1, n + 1):
    if i != v:
        print(v, "->", " -> ".join(map(str, path[i][1:])), "=", dist[i])

#4th Program

maxi = 9999999
n = int(input("Enter the number of nodes: "))
selected = [False for i in range(n)]
parent = [0 for i in range(n)]
def find(i):
    while parent[i] != i:
        i = parent[i]
    return i
def uni(i, j):
    x = find(i)
    y = find(j)
    parent[x] = y
def prim_mst(n, cost):
    n_edge = 0
    selected[0] = True
    while n_edge < n - 1:
        minimum = maxi
        x = y = 0
        for i in range(n):
            if selected[i]:
                for j in range(n):
                    if not selected[j] and cost[i][j]:
                        if minimum > cost[i][j]:
                            minimum = cost[i][j]
                            x = i
                            y = j
        print(x, '-->', y, ':', cost[x][y])
        selected[y] = True
        n_edge += 1
def kruskal_mst(n, cost):
    mincost = 0
    for i in range(n):
        parent[i] = i
    e_ctr = 0
    while e_ctr < n - 1:
        mini = maxi
        a = b = 0
        for i in range(n):
            for j in range(n):
                if cost[i][j] <= mini and find(i) != find(j):
                    mini = cost[i][j]
                    a = i
                    b = j
        uni(a, b)
        print("Edge", e_ctr + 1, ": (", a + 1, ",", b + 1, ") cost:", mini)
        e_ctr += 1
        mincost += mini
    print("Minimum cost =", mincost)
cost = [[int(x) for x in input().split()] for j in range(n)]
for i in range(n):
    for j in range(n):
        if cost[i][j] == 0:
            cost[i][j] = maxi
print("Minimum Spanning Tree using Prim's Algorithm:")
prim_mst(n, cost)
print("\nMinimum Spanning Tree using Kruskal's Algorithm:")
kruskal_mst(n, cost)
'''
#5th program 
import heapq
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq
def build_huffman_tree(characters, frequencies):
 # Create initial heap of Huffman nodes
    heap = []
    for i in range(len(characters)):
        node = HuffmanNode(characters[i], frequencies[i])
        heapq.heappush(heap, node)
    # Build Huffman tree by combining nodes until there is only one node left in the heap
    while len(heap) > 1:
        left_node = heapq.heappop(heap)
        right_node = heapq.heappop(heap)
        # Create a new node with combined frequency and left/right children
        combined_freq = left_node.freq + right_node.freq
        combined_node = HuffmanNode(None, combined_freq)
        combined_node.left = left_node
        combined_node.right = right_node
        heapq.heappush(heap, combined_node)
    return heapq.heappop(heap)
def build_huffman_codes(root, current_code, huffman_codes):
    if root is None:
        return
    # Leaf node represents a character
    if root.char is not None:
        huffman_codes[root.char] = current_code
        return
    # Traverse left branch (0) and append '0' to the current code
    build_huffman_codes(root.left, current_code + '0', huffman_codes)
    # Traverse right branch (1) and append '1' to the current code
    build_huffman_codes(root.right, current_code + '1', huffman_codes)
def encode_string(string, huffman_codes):
    encoded_string = ''
    for char in string:
        encoded_string += huffman_codes[char]
    return encoded_string
def decode_string(encoded_string, root):
    decoded_string = ''
    current_node = root
    for bit in encoded_string:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
    # Leaf node represents a character
        if current_node.char is not None:
            decoded_string += current_node.char
            current_node = root
    return decoded_string
# Get inputs from the user
n = int(input("Enter the number of characters: "))
characters = []
frequencies = []
for i in range(n):
    char = input("Enter character: ")
    frequency = float(input("Enter probability: "))
    characters.append(char)
    frequencies.append(frequency)
# Build Huffman tree
huffman_tree = build_huffman_tree(characters, frequencies)
# Build Huffman codes
huffman_codes = {}
build_huffman_codes(huffman_tree, '', huffman_codes)
# Encode and decode strings
string = input("Enter a string to encode: ")
encoded_string = encode_string(string, huffman_codes)
print("Huffman Codes:")
for char, code in huffman_codes.items():
    print(char, ":", code)
print("Encoded string:", encoded_string)
# Decode a Huffman-encoded string
decode_choice = input("Do you want to decode the encoded string? (Y/N): ")
if decode_choice.lower() == 'y':
    decoded_string = decode_string(encoded_string, huffman_tree)
    print("Decoded string:", decoded_string)

















