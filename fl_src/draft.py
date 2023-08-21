import numpy as np
import time
import matplotlib.pyplot as plt

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

def bubble_sort(arr):
    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True

        if not swapped:
            break

def main():
    
    lengths = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    insertion_sort_times = []
    bubble_sort_times = []

    
    for length in lengths:
        
        # change the random array type to int16 instead of float64.
        # this will make the sorting faster because each element will take up less memory
        # an int16 takes up 2 bytes (16 bits) of memory, while a float64 takes up 8 bytes (16 bits) of memory
        random_array = np.random.random(length).astype(np.int16) 
        start_time = time.time()
        insertion_sort(random_array)
        end_time = time.time()
        elapsed_time_insertion = end_time - start_time
        insertion_sort_times.append(elapsed_time_insertion)

        random_array = np.random.random(length).astype(np.int16)
        start_time = time.time()
        bubble_sort(random_array)
        end_time = time.time()
        elapsed_time_bubble = end_time - start_time
        bubble_sort_times.append(elapsed_time_bubble)


    plt.plot(lengths, insertion_sort_times, marker='o', label='Insertion Sort')
    plt.plot(lengths, bubble_sort_times, marker='o', label='Bubble Sort')
    plt.xlabel('Array Length')
    plt.ylabel('Sorting Time (seconds)')
    plt.title('Sorting Algorithm Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()